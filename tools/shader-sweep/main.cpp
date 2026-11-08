// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include "common/cityhash.h"
#include <nlohmann/json.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "video_core/pipeline_cache_files.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/frontend/ampere/trap_guards.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

namespace {
using Json = nlohmann::json;
using Shader::Stage;
using Shader::Ampere::Instruction;

struct MissingContext : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr std::array StageNames{"vertex",  "tess-control", "tess-eval", "geometry", "fragment",
                                "compute", "vertex-a",     "task",      "mesh"};

Stage ParseStage(std::string_view name) {
    for (size_t i = 0; i < StageNames.size(); ++i) {
        if (name == StageNames[i])
            return static_cast<Stage>(i);
    }
    throw std::runtime_error("Unknown stage: " + std::string{name});
}

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    const auto size = file.tellg();
    if (size < 0 || size > 64 * 1024 * 1024) {
        throw std::runtime_error("Input exceeds the 64 MiB limit");
    }
    std::vector<u8> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return bytes;
}

class Reader {
public:
    explicit Reader(std::span<const u8> bytes_) : bytes{bytes_} {}

    template <typename T>
    T Read() {
        T value;
        Copy(&value, sizeof(value));
        return value;
    }

    void Copy(void* destination, size_t size) {
        if (size > bytes.size())
            throw std::runtime_error("Truncated environment record");
        std::memcpy(destination, bytes.data(), size);
        bytes = bytes.subspan(size);
    }

    size_t Remaining() const {
        return bytes.size();
    }

private:
    std::span<const u8> bytes;
};

template <typename K, typename V>
V Lookup(const std::map<K, V>& values, K key, std::string_view name) {
    const auto it = values.find(key);
    if (it == values.end()) {
        throw MissingContext(fmt::format("Uncaptured {} at {:#x}", name, key));
    }
    return it->second;
}

u64 CbufKey(u32 bank, u32 offset) {
    return (u64{bank} << 32) | offset;
}

u64 IndirectCbufKey(u32 bank, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (u64{indirect & 0x3fffff} << 40) | CbufKey(bank, offset);
}

// --guess-missing: values used for reads the environment has no record of. The defaults
// equal the emulator's warm-up loader guesses.
struct Guesses {
    bool enabled{};
    u32 cbuf0{0x10000000};
    u32 cbuf_other{0};
    u32 texture_type{3};
    u32 pixel_format{16};
    bool storage_pointer{true};       // bank 0
    bool storage_pointer_other{false}; // banks 1..17 (GuessMissing::storage_pointer_other)
};

class OfflineEnvironment final : public Shader::Environment {
public:
    Guesses guesses;
    u32 guessed_reads{};

    void Load(std::span<const u8> bytes, u32 version) {
        Reader reader{bytes};
        const u64 size = reader.Read<u64>();
        const u64 type_count = reader.Read<u64>();
        const u64 format_count = reader.Read<u64>();
        const u64 cbuf_count = reader.Read<u64>();
        const u64 pointer_count = version >= 47 ? reader.Read<u64>() : 0;
        const u64 replacement_count = reader.Read<u64>();
        if (size % 8 || !size || size > reader.Remaining()) {
            throw std::runtime_error("Invalid environment code size");
        }
        for (const u64 count :
             {type_count, format_count, cbuf_count, pointer_count, replacement_count}) {
            if (count > bytes.size() / 8)
                throw std::runtime_error("Invalid environment entry count");
        }
        local_memory = reader.Read<u32>();
        texture_bound = reader.Read<u32>();
        start_address = reader.Read<u32>();
        low = reader.Read<u32>();
        high = reader.Read<u32>();
        viewport = reader.Read<u32>();
        const u32 stage_value = reader.Read<u32>();
        if (stage_value >= StageNames.size())
            throw std::runtime_error("Invalid cached stage");
        stage = static_cast<Stage>(stage_value);
        code.resize(size / 8);
        reader.Copy(code.data(), size);
        if (low % 8 || high < low || u64{high} - low + 8 != size) {
            throw std::runtime_error("Invalid cached instruction bounds");
        }
        for (u64 i = 0; i < type_count; ++i) {
            const u32 key = reader.Read<u32>();
            const u32 type = reader.Read<u32>();
            if (type >= Shader::NUM_TEXTURE_TYPES)
                throw std::runtime_error("Invalid texture type");
            texture_types.emplace(key, static_cast<Shader::TextureType>(type));
        }
        for (u64 i = 0; i < format_count; ++i) {
            const u32 key = reader.Read<u32>();
            const u32 value = reader.Read<u32>();
            if (value > std::numeric_limits<u8>::max()) {
                throw std::runtime_error("Invalid texture pixel format");
            }
            texture_formats.emplace(key, static_cast<Shader::TexturePixelFormat>(value));
        }
        for (u64 i = 0; i < cbuf_count; ++i) {
            const u64 key = reader.Read<u64>();
            cbuf_values.emplace(key, reader.Read<u32>());
        }
        for (u64 i = 0; i < pointer_count; ++i) {
            const u64 key = reader.Read<u64>();
            const u8 value = reader.Read<u8>();
            if (value > 1)
                throw std::runtime_error("Invalid storage pointer classification");
            pointers.emplace(key, value);
        }
        for (u64 i = 0; i < replacement_count; ++i) {
            const u64 key = reader.Read<u64>();
            const u32 value = reader.Read<u32>();
            if (value > 2)
                throw std::runtime_error("Invalid constant replacement");
            replacements.emplace(key, static_cast<Shader::ReplaceConstant>(value));
        }
        if (stage == Stage::Compute) {
            workgroup = reader.Read<std::array<u32, 3>>();
            shared_memory = reader.Read<u32>();
            if (!workgroup[0] || !workgroup[1] || !workgroup[2]) {
                throw std::runtime_error("Invalid cached workgroup dimensions");
            }
        } else {
            sph = reader.Read<Shader::ProgramHeader>();
            if (stage == Stage::Geometry)
                gp_passthrough_mask = reader.Read<std::array<u32, 8>>();
            if (stage == Stage::Task || stage == Stage::Mesh) {
                if (version < 53)
                    throw std::runtime_error("Task/mesh environments require cache version 53");
                mesh_registers = reader.Read<std::array<u32, 3>>();
                mesh_primitive_header = reader.Read<Shader::ProgramHeader>();
            }
        }
        if (reader.Remaining())
            throw std::runtime_error("Trailing bytes after environment record");
        is_proprietary_driver = texture_bound == 2 || IsNativeProgram();
        captured = true;
    }

    void SetStage(Stage value) {
        stage = value;
    }

    void SetHeader(std::span<const u8> bytes) {
        std::memcpy(&sph, bytes.data(), sizeof(sph));
        const u64 size = sph.LocalMemorySize() + sph.common3.shader_local_memory_crs_size;
        if (size > std::numeric_limits<u32>::max())
            throw std::runtime_error("Local memory overflow");
        local_memory = static_cast<u32>(size);
    }

    void SetCode(const Shader::Ampere::LoweredProgram& lowered) {
        code.resize(sizeof(sph) / sizeof(u64));
        std::memcpy(code.data(), &sph, sizeof(sph));
        for (const u64 instruction : lowered.instructions) {
            if (code.size() % 4 == 0)
                code.push_back(0);
            code.push_back(instruction);
        }
        code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
        low = 0;
        high = static_cast<u32>(code.size() * 8 - 8);
        start_address = stage == Stage::Compute ? sizeof(sph) : 0;
        if (lowered.required_texture_buffer != ~0U)
            texture_bound = lowered.required_texture_buffer;
        is_proprietary_driver = true;
    }

    u32 EntryPoint() const {
        return start_address + (stage == Stage::Compute ? 0 : sizeof(sph));
    }

    u64 ReadInstruction(u32 address) override {
        if (address < low || address > high || (address - low) % 8) {
            throw std::runtime_error(
                fmt::format("Instruction address outside capture: {:#x}", address));
        }
        return code.at((address - low) / 8);
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (!IsNativeProgram())
            return address;
        const auto result = Shader::Ampere::NativeProgramAddress(code, address);
        if (!result)
            throw std::runtime_error("Invalid native indirect branch target");
        return *result;
    }
    bool IsNativeProgram() const override {
        return Shader::Ampere::HasNativeAddressMap(code);
    }
    bool HasNativeInstructions() const override {
        return IsNativeProgram();
    }
    /// Recorded storage pointer classification, or the bank's guess when guessing is on.
    bool IsCbufStorageBufferPointer(u32 bank, u32 offset) override {
        if (static_context && !pointers.contains(CbufKey(bank, offset)) && bank == 0 &&
            offset % 16 == 0) {
            // Bank 0 holds each stage's storage buffer bindings (address, size) at 16-byte
            // slots: a program reading one uses a bound storage buffer.
            pointers.emplace(CbufKey(bank, offset), 1);
            return true;
        }
        if (static_context && !pointers.contains(CbufKey(bank, offset)) && bank == 1) {
            // Bank 1 is the program's own constant table (jump targets and literals compiled
            // into the shader), never a GPU address.
            pointers.emplace(CbufKey(bank, offset), 0);
            return false;
        }
        return Find(pointers, CbufKey(bank, offset), "storage pointer classification",
                    u8{bank == 0 ? guesses.storage_pointer : guesses.storage_pointer_other}) != 0;
    }
    /// Recorded storage pointer table classification, or the bank's guess when guessing is on.
    bool IsCbufIndirectStorageBufferPointer(u32 bank, u32 offset, u32 indirect) override {
        return Find(pointers, IndirectCbufKey(bank, offset, indirect),
                    "storage pointer table classification",
                    u8{bank == 0 ? guesses.storage_pointer : guesses.storage_pointer_other}) != 0;
    }
    /// Recorded constant buffer value, or the bank's guess when guessing is on.
    u32 ReadCbufValue(u32 bank, u32 offset) override {
        if (static_context && bank == 1 && !cbuf_values.contains(CbufKey(bank, offset))) {
            const u32 value =
                Lookup(static_driver_values, CbufKey(bank, offset), "driver constant (static)");
            cbuf_values.emplace(CbufKey(bank, offset), value);
            return value;
        }
        return Find(cbuf_values, CbufKey(bank, offset), "constant buffer value",
                    bank == 0 ? guesses.cbuf0 : guesses.cbuf_other);
    }
    /// A texture handle word: with the static context, a placeholder handle (translation needs
    /// only the bound texture's type, see ReadTextureType below).
    u32 ReadTextureHandleWord(u32 bank, u32 offset) override {
        if (static_context && (bank == StaticTextureBank || stage == Stage::Compute) &&
            bank != 1 && !cbuf_values.contains(CbufKey(bank, offset))) {
            const u32 handle = StaticHandleBase | (offset / 4);
            cbuf_values.emplace(CbufKey(bank, offset), handle);
            return handle;
        }
        return ReadCbufValue(bank, offset);
    }
    /// Driver bank 1 holds the program's constant table: the code bytes from the offset in
    /// the control section's word at byte 64 to the end of the allocation.
    bool SetStaticContext(std::span<const u8> allocation, std::span<const u8> control) {
        static_context = true;
        if (control.size() < 68)
            return false;
        u32 table{};
        std::memcpy(&table, control.data() + 64, sizeof(table));
        if (table % 4 || table >= allocation.size())
            return false; // no constant table: bank 1 reads stay uncaptured
        for (size_t at = table; at + 4 <= allocation.size(); at += 4) {
            u32 word{};
            std::memcpy(&word, allocation.data() + at, sizeof(word));
            static_driver_values.emplace(CbufKey(1, static_cast<u32>(at - table)), word);
        }
        return true;
    }
    /// A compute program's launch parameters sit in its control section: workgroup size at
    /// bytes 88..99, shared memory at 100 (the driver allocates it in 256-byte units) and
    /// local memory at 104. Its texture handles are read from bank 0.
    void SetStaticCompute(std::span<const u8> control) {
        if (control.size() < 108)
            throw std::runtime_error("Compute control section is too short");
        std::array<u32, 5> words{};
        std::memcpy(words.data(), control.data() + 88, sizeof(words));
        if (!words[0] || !words[1] || !words[2] || words[0] * words[1] * words[2] > 1024)
            throw std::runtime_error("Invalid compute workgroup in control section");
        workgroup = {words[0], words[1], words[2]};
        shared_memory = (words[3] + 255) & ~255U;
        local_memory = words[4];
        if (texture_bound == ~0U)
            texture_bound = 0;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        if (static_context && !texture_types.contains(handle)) {
            // The instruction's own type decides 2D vs array; this answer only separates
            // rect and buffer textures, which Ocarina's captures never use.
            texture_types.emplace(handle, Shader::TextureType::Color2D);
            return Shader::TextureType::Color2D;
        }
        return Find(texture_types, handle, "texture type",
                    static_cast<Shader::TextureType>(guesses.texture_type));
    }
    bool CanReadTextureDescriptors() const override {
        return !static_context;
    }
    void NoteInferredTextureType(u32 handle, Shader::TextureType type) override {
        if (!static_context)
            return;
        const auto [it, inserted] = texture_types.emplace(handle, type);
        if (!inserted && it->second != type)
            throw MissingContext("Texture handle is used with conflicting types");
    }
    /// The record Lichii's pipeline cache stores for this environment
    /// (GenericEnvironment::Serialize).
    std::vector<u8> Serialize() const {
        std::vector<u8> out;
        const auto put = [&](const auto& value) {
            const auto* bytes = reinterpret_cast<const u8*>(&value);
            out.insert(out.end(), bytes, bytes + sizeof(value));
        };
        put(u64{code.size() * sizeof(u64)});
        put(u64{texture_types.size()});
        put(u64{texture_formats.size()});
        put(u64{cbuf_values.size()});
        put(u64{pointers.size()});
        put(u64{replacements.size()});
        put(local_memory);
        put(texture_bound);
        put(start_address);
        put(low);
        put(high);
        put(viewport);
        put(static_cast<u32>(stage));
        const auto* words = reinterpret_cast<const u8*>(code.data());
        out.insert(out.end(), words, words + code.size() * sizeof(u64));
        for (const auto& [key, type] : texture_types) {
            put(key);
            put(static_cast<u32>(type));
        }
        for (const auto& [key, format] : texture_formats) {
            put(key);
            put(static_cast<u32>(format));
        }
        for (const auto& [key, value] : cbuf_values) {
            put(key);
            put(value);
        }
        for (const auto& [key, mapped] : pointers) {
            put(key);
            put(mapped);
        }
        for (const auto& [key, value] : replacements) {
            put(key);
            put(static_cast<u32>(value));
        }
        if (stage == Stage::Compute) {
            put(workgroup);
            put(shared_memory);
        } else {
            put(sph);
        }
        return out;
    }
    std::array<u32, 3> Workgroup() const {
        return workgroup;
    }
    u32 SharedMemory() const {
        return shared_memory;
    }
    u64 assumed_formats{};
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        if (static_context && !texture_formats.contains(handle)) {
            // Ocarina's captures bind no integer textures or images (0 of 150 compute programs).
            ++assumed_formats;
            texture_formats.emplace(handle, Shader::TexturePixelFormat::A8B8G8R8_UNORM);
            return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
        }
        return Find(texture_formats, handle, "texture pixel format",
                    static_cast<Shader::TexturePixelFormat>(guesses.pixel_format));
    }
    bool IsTexturePixelFormatInteger(u32 handle) override {
        using F = Shader::TexturePixelFormat;
        switch (ReadTexturePixelFormat(handle)) {
        case F::A8B8G8R8_SINT:
        case F::A8B8G8R8_UINT:
        case F::A2B10G10R10_UINT:
        case F::R8_SINT:
        case F::R8_UINT:
        case F::R16G16B16A16_SINT:
        case F::R16G16B16A16_UINT:
        case F::R32G32B32A32_UINT:
        case F::R32G32B32A32_SINT:
        case F::R32G32_SINT:
        case F::R16_UINT:
        case F::R16_SINT:
        case F::R16G16_UINT:
        case F::R16G16_SINT:
        case F::R8G8_SINT:
        case F::R8G8_UINT:
        case F::R32G32_UINT:
        case F::R32_UINT:
        case F::R32_SINT:
            return true;
        default:
            return false;
        }
    }
    u32 ReadViewportTransformState() override {
        return viewport;
    }
    u32 TextureBoundBuffer() const override {
        if (texture_bound == ~0U && static_context)
            return StaticTextureBank;
        if (texture_bound == ~0U)
            throw MissingContext("Uncaptured texture-bound constant buffer");
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return local_memory;
    }
    u32 SharedMemorySize() const override {
        return shared_memory;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return workgroup;
    }
    bool HasHLEMacroState() const override {
        return !replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = replacements.find(CbufKey(bank, offset));
        return it == replacements.end() ? std::nullopt : std::optional{it->second};
    }
    void Dump(u64, u64) override {}

    bool captured{};
    std::vector<u64> code;

private:
    /// The recorded value for `key`. With guessing on, a missing value counts as a guessed read and
    /// returns `guess`; otherwise it throws.
    template <typename K, typename V>
    V Find(const std::map<K, V>& values, K key, std::string_view name, V guess) {
        if (guesses.enabled && !values.contains(key)) {
            ++guessed_reads;
            return guess;
        }
        return Lookup(values, key, name);
    }

    u32 low{}, high{}, local_memory{}, shared_memory{}, viewport{1}, texture_bound{~0U};
    std::array<u32, 3> workgroup{1, 1, 1};
    std::map<u32, Shader::TextureType> texture_types;
    std::map<u32, Shader::TexturePixelFormat> texture_formats;
    std::map<u64, u32> cbuf_values;
    std::map<u64, u8> pointers;
    std::map<u64, Shader::ReplaceConstant> replacements;
    std::map<u64, u32> static_driver_values;
    bool static_context{};
    static constexpr u32 StaticTextureBank = 2;
    static constexpr u32 StaticHandleBase = 0x00F00000;
};

template <typename T>
void WriteFile(const std::filesystem::path& path, const std::vector<T>& words) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file ||
        !file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(T))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    file.close();
    if (!file)
        throw std::runtime_error("Cannot close " + path.string());
}

Shader::RuntimeInfo Runtime(std::span<const u8> key, Stage stage, Json& assumptions) {
    Shader::RuntimeInfo info{};
    info.previous_stage_stores.mask.set();
    assumptions.push_back(
        "Sibling stage linkage is unavailable; all previous-stage varyings are enabled");
    if (key.empty()) {
        assumptions.push_back("No pipeline key: point input topology, isolines/equal tessellation, "
                              "float attributes, no alpha test or transform feedback");
        return info;
    }
    if (key.size() != 1056 || stage == Stage::Compute) {
        throw std::runtime_error("Expected a 1056-byte graphics pipeline key");
    }
    const auto word = [&](size_t offset) {
        u32 value;
        std::memcpy(&value, key.data() + offset, sizeof(value));
        return value;
    };
    const u32 raw1 = word(48), raw2 = word(52);
    if (raw1 & (1U << 6))
        throw MissingContext("Transform feedback pipeline state is not supported offline");
    const u32 topology = (raw1 >> 24) & 15;
    switch (topology) {
    case 0:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case 1:
    case 2:
    case 3:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 14:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case 10:
    case 11:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case 12:
    case 13:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    default:
        throw std::runtime_error("Invalid pipeline input topology");
    }
    if (stage == Stage::TessellationEval) {
        const u32 primitive = (raw1 >> 10) & 3, spacing = (raw1 >> 12) & 3;
        if (primitive > 2 || spacing > 2)
            throw std::runtime_error("Invalid tessellation state");
        info.tess_primitive = static_cast<Shader::TessPrimitive>(primitive);
        info.tess_spacing = static_cast<Shader::TessSpacing>(spacing);
        info.tess_clockwise = (raw1 >> 14) & 1;
    }
    info.force_early_z = (raw2 >> 4) & 1;
    info.y_negate = (raw2 >> 11) & 1;
    if (stage == Stage::Geometry || (stage == Stage::VertexB && word(32) == 0 && word(36) == 0)) {
        info.convert_depth_mode = (raw1 >> 7) & 1;
        if (topology == 0)
            info.fixed_state_point_size = std::bit_cast<float>(word(76));
    }
    if (stage == Stage::Fragment) {
        info.alpha_test_func = static_cast<Shader::CompareFunction>((raw2 >> 1) & 7);
        info.alpha_test_reference = std::bit_cast<float>(word(72));
        info.dual_source_blend = (raw1 >> 21) & 1;
        for (size_t index = 0; index < info.color_output_types.size(); ++index) {
            switch (key[56 + index]) {
            case 0xc1:
            case 0xc4:
            case 0xc8:
            case 0xcc:
            case 0xd8:
            case 0xdc:
            case 0xe3:
            case 0xec:
            case 0xf0:
            case 0xf5:
                info.color_output_types[index] = Shader::AttributeType::SignedInt;
                break;
            case 0xc2:
            case 0xc5:
            case 0xc9:
            case 0xcd:
            case 0xd2:
            case 0xd9:
            case 0xdd:
            case 0xe4:
            case 0xed:
            case 0xf1:
            case 0xf6:
                info.color_output_types[index] = Shader::AttributeType::UnsignedInt;
                break;
            case 0:
            case 0xc0:
            case 0xc3:
            case 0xc6:
            case 0xc7:
            case 0xca:
            case 0xcb:
            case 0xce:
            case 0xcf:
            case 0xd0:
            case 0xd1:
            case 0xd5:
            case 0xd6:
            case 0xd7:
            case 0xda:
            case 0xdb:
            case 0xde:
            case 0xdf:
            case 0xe0:
            case 0xe5:
            case 0xe6:
            case 0xe7:
            case 0xe8:
            case 0xe9:
            case 0xea:
            case 0xeb:
            case 0xee:
            case 0xef:
            case 0xf2:
            case 0xf3:
            case 0xf4:
            case 0xf8:
            case 0xf9:
            case 0xfa:
                info.color_output_types[index] = Shader::AttributeType::Float;
                break;
            default:
                throw std::runtime_error("Invalid pipeline color attachment format");
            }
        }
    }
    if (stage == Stage::VertexB) {
        for (size_t index = 0; index < info.generic_input_types.size(); ++index) {
            auto& type = info.generic_input_types[index];
            if (raw1 & (1U << 5)) {
                constexpr std::array types{
                    Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                    Shader::AttributeType::SignedInt, Shader::AttributeType::UnsignedInt};
                type = types[(word(112 + (index / 16) * 4) >> ((index % 16) * 2)) & 3];
                continue;
            }
            const u32 attribute = word(160 + index * 4);
            if (!(attribute & 1)) {
                type = Shader::AttributeType::Disabled;
                continue;
            }
            switch ((attribute >> 20) & 7) {
            case 1:
            case 2:
            case 7:
                type = Shader::AttributeType::Float;
                break;
            case 3:
                type = Shader::AttributeType::SignedInt;
                break;
            case 4:
                type = Shader::AttributeType::UnsignedInt;
                break;
            case 5:
                type = Shader::AttributeType::UnsignedScaled;
                break;
            case 6:
                type = Shader::AttributeType::SignedScaled;
                break;
            default:
                throw std::runtime_error("Invalid pipeline vertex attribute type");
            }
        }
    }
    assumptions.push_back("Pipeline key supplies vertex/fragment interface types and fixed state; "
                          "integer fragment attachments use typed outputs on every offline host");
    return info;
}
} // namespace

/// Parses the command line, reads the program (and its environment and pipeline key when given),
/// translates it and writes the SPIR-V or lowered code. Prints a JSON report; the exit code is 2
/// for bad input and 1 for a failed translation.
int main(int argc, char** argv) {
    Json diagnostic{{"schema_version", 1}, {"status", "error"},           {"phase", "input"},
                    {"stage", nullptr},    {"input_format", nullptr},     {"source_bytes", 0},
                    {"spirv_words", 0},    {"assumptions", Json::array()}};
    try {
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error("Only little-endian hosts are supported");
        }
        std::string input, output, environment, pipeline_key, control, allocation, cache_record,
            override_out,
            format{"instructions"};
        std::optional<Stage> requested_stage;
        u32 version{};
        bool lower_only{};
        bool packed_code{};
        bool capture_extent{};
        Guesses guesses;
        bool storage_pointer_other_set{};
        for (int i = 1; i < argc; ++i) {
            const std::string arg{argv[i]};
            if (arg == "--help") {
                std::cout << "shader-sweep [INPUT] [--stage "
                             "vertex|tess-control|tess-eval|geometry|fragment|compute|task|mesh]\n"
                             "  [--format instructions|native-header] [--environment FILE "
                             "--cache-version 46|47|48|49|50|51|52|53|54]\n"
                             "  [--guess-missing [--guess-cbuf0 HEX] [--guess-cbuf-other HEX]\n"
                             "   [--guess-texture-type N] [--guess-pixel-format N] "
                             "[--guess-storage-pointer 0|1]\n"
                             "   [--guess-storage-pointer-other 0|1]]\n"
                             "  [--control FILE [--allocation FILE] [--cache-record FILE]]\n"
                             "  [--pipeline-key FILE] [--capture-extent] [--override-out FILE] "
                             "(--output SPV | --lower-only WORDS | --packed-code WORDS)\n";
                return 0;
            }
            if (!arg.starts_with("--")) {
                if (!input.empty())
                    throw std::runtime_error("Multiple input files");
                input = arg;
                continue;
            }
            if (arg == "--capture-extent") {
                capture_extent = true;
                continue;
            }
            if (arg == "--guess-missing") {
                guesses.enabled = true;
                continue;
            }
            if (++i == argc)
                throw std::runtime_error("Missing value for " + arg);
            const std::string value{argv[i]};
            if (arg == "--stage")
                requested_stage = ParseStage(value);
            else if (arg == "--format")
                format = value;
            else if (arg == "--environment")
                environment = value;
            else if (arg == "--override-out")
                override_out = value;
            else if (arg == "--pipeline-key")
                pipeline_key = value;
            else if (arg == "--control")
                control = value;
            else if (arg == "--allocation")
                allocation = value;
            else if (arg == "--cache-record")
                cache_record = value;
            else if (arg == "--guess-cbuf0")
                guesses.cbuf0 = static_cast<u32>(std::stoul(value, nullptr, 16));
            else if (arg == "--guess-cbuf-other")
                guesses.cbuf_other = static_cast<u32>(std::stoul(value, nullptr, 16));
            else if (arg == "--guess-texture-type") {
                guesses.texture_type = static_cast<u32>(std::stoul(value, nullptr, 0));
                if (guesses.texture_type >= Shader::NUM_TEXTURE_TYPES)
                    throw std::runtime_error("Invalid --guess-texture-type");
            } else if (arg == "--guess-pixel-format") {
                guesses.pixel_format = static_cast<u32>(std::stoul(value, nullptr, 0));
                if (guesses.pixel_format > std::numeric_limits<u8>::max())
                    throw std::runtime_error("Invalid --guess-pixel-format");
            } else if (arg == "--guess-storage-pointer") {
                if (value != "0" && value != "1")
                    throw std::runtime_error("Expected --guess-storage-pointer 0 or 1");
                guesses.storage_pointer = value == "1";
                if (!storage_pointer_other_set)
                    guesses.storage_pointer_other = guesses.storage_pointer;
            } else if (arg == "--guess-storage-pointer-other") {
                if (value != "0" && value != "1")
                    throw std::runtime_error("Expected --guess-storage-pointer-other 0 or 1");
                guesses.storage_pointer_other = value == "1";
                storage_pointer_other_set = true;
            } else if (arg == "--cache-version") {
                if (value != "46" && value != "47" && value != "48" && value != "49" && value != "50" &&
                    value != "51" && value != "52" && value != "53" && value != "54")
                    throw std::runtime_error("Expected cache version 46 to 54");
                version = static_cast<u32>(std::stoul(value));
            } else if (arg == "--output" || arg == "--lower-only" || arg == "--packed-code") {
                if (!output.empty())
                    throw std::runtime_error("Only one output mode is permitted");
                output = value;
                lower_only = arg != "--output";
                packed_code = arg == "--packed-code";
            } else
                throw std::runtime_error("Unknown option " + arg);
        }
        if (output.empty() || (input.empty() && environment.empty()) ||
            (environment.empty() != (version == 0)) || (lower_only && input.empty()) ||
            (capture_extent && input.empty())) {
            throw std::runtime_error("Invalid arguments; see --help");
        }
        if (format != "instructions" && format != "native-header")
            throw std::runtime_error("Invalid input format");
        for (const auto& path : {input, environment, pipeline_key}) {
            if (!path.empty() && std::filesystem::weakly_canonical(path) ==
                                     std::filesystem::weakly_canonical(output)) {
                throw std::runtime_error("Output must not overwrite an input");
            }
        }
        OfflineEnvironment env;
        env.guesses = guesses;
        if (!environment.empty())
            env.Load(ReadFile(environment), version);
        if (requested_stage) {
            if (env.captured && env.ShaderStage() != *requested_stage)
                throw std::runtime_error("Stage disagrees with environment");
            env.SetStage(*requested_stage);
        }
        diagnostic["input_format"] = input.empty() ? "environment" : format;
        if (!input.empty()) {
            const auto bytes = ReadFile(input);
            const size_t header_size = format == "native-header" ? 128 : 0;
            if (bytes.size() <= header_size || (bytes.size() - header_size) % sizeof(Instruction) ||
                (bytes.size() - header_size) / sizeof(Instruction) > 65536) {
                throw std::runtime_error(
                    "Native input must contain 1..65536 complete 16-byte instructions");
            }
            if (header_size) {
                Shader::ProgramHeader header;
                std::memcpy(&header, bytes.data(), sizeof(header));
                const u32 type = header.common0.shader_type;
                if (type < 1 || type > 5)
                    throw std::runtime_error("Invalid native graphics header stage");
                const Stage header_stage = static_cast<Stage>(type - 1);
                const Stage stage = requested_stage.value_or(
                    env.captured ? env.ShaderStage() : header_stage);
                const bool mesh_envelope =
                    (stage == Stage::Task && type == 1) ||
                    (stage == Stage::Mesh && (type == 1 || type == 3));
                if ((stage != header_stage && !mesh_envelope) ||
                    (env.captured && env.ShaderStage() != stage)) {
                    throw std::runtime_error("Stage disagrees with native header");
                }
                env.SetStage(stage);
                env.SetHeader(bytes);
                if (!control.empty()) {
                    diagnostic["static_context"] = env.SetStaticContext(bytes, ReadFile(control));
                }
            } else if (!control.empty() && env.ShaderStage() == Stage::Compute) {
                // The constant table lies past the declared code: read it from the allocation.
                diagnostic["static_context"] = env.SetStaticContext(
                    allocation.empty() ? bytes : ReadFile(allocation), ReadFile(control));
                env.SetStaticCompute(ReadFile(control));
            } else if (!requested_stage && !env.captured && !lower_only) {
                throw std::runtime_error(
                    "Instruction-only input requires --stage or a captured environment");
            }
            if (requested_stage || env.captured || header_size) {
                diagnostic["stage"] = StageNames[static_cast<u32>(env.ShaderStage())];
            }
            diagnostic["source_bytes"] = bytes.size() - header_size;
            std::vector<Instruction> native((bytes.size() - header_size) / sizeof(Instruction));
            std::memcpy(native.data(), bytes.data() + header_size,
                        native.size() * sizeof(Instruction));
            if (capture_extent) {
                diagnostic["phase"] = "capture";
                diagnostic["allocation_bytes"] = native.size() * sizeof(Instruction);
                Shader::Ampere::ProgramCaptureState capture;
                bool complete = false;
                for (u32 index = 0; index < native.size(); ++index) {
                    const auto result = capture.Observe(native[index], index);
                    if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
                        throw MissingContext("Native allocation has an out-of-range branch target");
                    }
                    if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
                        native.resize(index + 1);
                        complete = true;
                        break;
                    }
                }
                if (!complete) {
                    throw MissingContext("Native allocation lacks a complete runtime capture");
                }
                diagnostic["source_bytes"] = native.size() * sizeof(Instruction);
            }
            diagnostic["phase"] = "lower";
            std::vector<u32> traps;
            for (u32 i = 0; i < native.size(); ++i) {
                if (native[i].Bits(0, 12) == 0x95c)
                    traps.push_back(i * sizeof(Instruction));
            }
            if (!traps.empty()) {
                if (!env.captured)
                    throw MissingContext("Trap guards require captured constant buffers");
                const bool verified = env.ShaderStage() == Stage::Compute &&
                                      Shader::Ampere::VerifyTrapGuards(
                                          std::span{native}.first(traps.back() / 16 + 1),
                                          [&](u32 bank, u32 offset) -> std::optional<u32> {
                                              return env.ReadCbufValue(bank, offset);
                                          });
                if (!verified)
                    throw MissingContext("Native trap guards could not be verified");
            }
            const auto tessellation = env.ShaderStage() == Stage::TessellationControl
                                          ? Shader::Ampere::TessellationStage::Control
                                      : env.ShaderStage() == Stage::TessellationEval
                                          ? Shader::Ampere::TessellationStage::Evaluation
                                          : Shader::Ampere::TessellationStage::None;
            const auto lowered = Shader::Ampere::LowerToMaxwell(
                native, env.ShaderStage() == Stage::Geometry, true, traps, tessellation);
            if (!lowered) {
                diagnostic["error"] = {{"code", "lowering_failed"},
                                       {"message", lowered.error},
                                       {"offset", lowered.error_offset}};
                std::cout << diagnostic.dump() << '\n';
                return 1;
            }
            diagnostic["lowered_words"] = lowered.instructions.size();
            // The emulator's key for this program (Shader::Ampere::NativeProgramKey over the
            // captured source), and the lowering as a program override record
            // (the tools/nx2/make_*_warm.py scripts read it).
            diagnostic["native_key"] = fmt::format("{:016x}", Shader::Ampere::NativeProgramKey(native));
            if (!override_out.empty()) {
                const auto record = Shader::Ampere::EncodeProgramOverride(native, lowered);
                if (record.empty())
                    throw std::runtime_error("Cannot encode the lowering as an override record");
                WriteFile(override_out, record);
            }
            if (lower_only) {
                diagnostic["phase"] = "output";
                if (packed_code) {
                    env.SetCode(lowered);
                    WriteFile(output, env.code);
                } else {
                    WriteFile(output, lowered.instructions);
                }
                diagnostic["status"] = "ok";
                std::cout << diagnostic.dump() << '\n';
                return 0;
            }
            if (!env.captured && !header_size && env.ShaderStage() != Stage::Compute) {
                throw MissingContext(
                    "Graphics compilation requires a native header or captured environment");
            }
            env.SetCode(lowered);
            if (!env.captured) {
                diagnostic["assumptions"].push_back(
                    "No environment: workgroup 1x1x1, shared memory 0, viewport enabled, no HLE "
                    "replacements");
                if (!header_size)
                    diagnostic["assumptions"].push_back(
                        "Compute local memory defaults to zero without a captured environment");
            }
        }
        diagnostic["stage"] = StageNames[static_cast<u32>(env.ShaderStage())];
        if (env.ShaderStage() == Stage::Task || env.ShaderStage() == Stage::Mesh ||
            env.ShaderStage() == Stage::VertexA) {
            throw MissingContext(
                "Task, mesh and vertex-A require additional pipeline stage context");
        }
        const auto key = pipeline_key.empty() ? std::vector<u8>{} : ReadFile(pipeline_key);
        auto runtime = Runtime(key, env.ShaderStage(), diagnostic["assumptions"]);
        diagnostic["assumptions"].push_back(
            "Offline SPIR-V 1.4 profile; no device feature probing, default shader settings, no "
            "resolution scaling");
        const auto compile = [&](OfflineEnvironment& source) {
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
            Shader::Maxwell::Flow::CFG cfg{source, flow_pool, Shader::Maxwell::Location{source.EntryPoint()}};
            Shader::ObjectPool<Shader::IR::Inst> inst_pool;
            Shader::ObjectPool<Shader::IR::Block> block_pool;
            Shader::HostTranslateInfo host{};
            host.support_float64 = true;
            host.support_float16 = true;
            host.support_int64 = true;
            host.support_conditional_barrier = true;
            host.support_viewport_index_layer = true;
            host.support_geometry_shader_passthrough = true;
            host.ApplyDescriptorLimitPolicy();
            auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, source, cfg, host);
            auto info = runtime;
            Shader::Maxwell::ConvertLegacyToGeneric(program, info);
            Shader::Profile profile{};
            profile.supported_spirv = 0x00010400;
            profile.unified_descriptor_binding = true;
            profile.min_ssbo_alignment = host.min_ssbo_alignment;
            profile.support_int64 = true;
            profile.support_int16 = true;
            profile.support_int8 = true;
            profile.support_vote = true;
            profile.support_shuffle_relative = true;
            profile.support_geometry_streams = true;
            profile.support_geometry_shader_passthrough = true;
            profile.support_viewport_index_layer_non_geometry = true;
            profile.support_multi_viewport = true;
            profile.max_user_clip_distances = 8;
            profile.max_subgroup_size = 32;
            profile.support_typeless_image_loads = true;
            Shader::Backend::Bindings bindings{};
            return Shader::Backend::SPIRV::EmitSPIRV(profile, info, program, bindings);
        };
        diagnostic["phase"] = "translate";
        diagnostic["native_hash"] =
            fmt::format("{:016x}", Common::CityHash64(reinterpret_cast<const char*>(env.code.data()),
                                                      env.code.size() * sizeof(u64)) ^
                                       0x4f554e4345000002ULL);
        const auto spirv = compile(env);
        if (!cache_record.empty()) {
            // The record Lichii loads at boot: it must reproduce this translation by itself.
            diagnostic["phase"] = "cache-record";
            if (env.ShaderStage() != Stage::Compute || env.captured)
                throw std::runtime_error("Cache records are written for static compute programs");
            const auto record_env = env.Serialize();
            OfflineEnvironment reloaded;
            reloaded.Load(record_env, VideoCommon::VulkanPipelineCacheVersion);
            if (compile(reloaded) != spirv)
                throw MissingContext("Cache record does not reproduce the static translation");
            const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(env.code.data()),
                                                env.code.size() * sizeof(u64)) ^
                             0x4f554e4345000002ULL;
            std::vector<u8> record(sizeof(u32));
            record[0] = 1; // one environment
            record.insert(record.end(), record_env.begin(), record_env.end());
            const auto put = [&](const auto& value) {
                const auto* bytes = reinterpret_cast<const u8*>(&value);
                record.insert(record.end(), bytes, bytes + sizeof(value));
            };
            put(hash);                // ComputePipelineCacheKey::unique_hash
            put(env.SharedMemory());  // shared_memory_size
            put(env.Workgroup());     // workgroup_size
            WriteFile(cache_record, record);
            diagnostic["cache_record"] = {{"unique_hash", fmt::format("{:016x}", hash)},
                                          {"assumed_pixel_formats", env.assumed_formats},
                                          {"bytes", record.size()}};
        }
        if (guesses.enabled)
            diagnostic["guessed_reads"] = env.guessed_reads;
        diagnostic["phase"] = "output";
        diagnostic["spirv_words"] = spirv.size();
        WriteFile(output, spirv);
        diagnostic["status"] = "ok";
        std::cout << diagnostic.dump() << '\n';
        return 0;
    } catch (const MissingContext& error) {
        diagnostic["phase"] = "context";
        diagnostic["error"] = {{"code", "missing_context"}, {"message", error.what()}};
    } catch (const std::exception& error) {
        diagnostic["error"] = {
            {"code", diagnostic["phase"] == "input" ? "invalid_input" : "compilation_failed"},
            {"message", error.what()}};
    }
    std::cout << diagnostic.dump() << '\n';
    return diagnostic["phase"] == "input" ? 2 : 1;
}
