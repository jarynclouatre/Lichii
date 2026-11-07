// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "video_core/renderer_vulkan/vk_shader_reuse.h"

namespace {

class TestEnvironment final : public Shader::Environment {
public:
    bool native{};
    bool descriptors{true};
    std::optional<Shader::TextureType> inferred_type;
    bool pointer{true};
    bool indirect_pointer{true};
    u32 cbuf_base{};
    bool throws{};

    TestEnvironment() {
        stage = Shader::Stage::Fragment;
    }
    bool IsNativeProgram() const override {
        return native;
    }
    bool HasNativeInstructions() const override {
        return native;
    }
    size_t CodeWords() const override {
        return 4096;
    }
    bool CanReadTextureDescriptors() const override {
        return descriptors;
    }
    void NoteInferredTextureType(u32, Shader::TextureType type) override {
        inferred_type = type;
    }
    bool IsCbufStorageBufferPointer(u32, u32) override {
        return pointer;
    }
    bool IsCbufIndirectStorageBufferPointer(u32, u32, u32) override {
        return indirect_pointer;
    }
    u64 ReadInstruction(u32) override {
        return 0;
    }
    u32 ReadCbufValue(u32, u32 offset) override {
        if (throws) {
            throw Shader::LogicError("Uncached read");
        }
        return cbuf_base + offset;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

/// What a native program's translation asks: a constant, then a storage buffer pointer slot and
/// the pointer table behind it.
Vulkan::TranslationReads Translate(TestEnvironment& env) {
    Vulkan::RecordingEnvironment recorder{env};
    REQUIRE(recorder.ReadCbufValue(1, 8) == env.cbuf_base + 8);
    REQUIRE(recorder.IsCbufStorageBufferPointer(1, 16) == env.pointer);
    REQUIRE(recorder.IsCbufIndirectStorageBufferPointer(1, 16, 8) == env.indirect_pointer);
    REQUIRE(recorder.IsNativeProgram() == env.native);
    // Asked twice, recorded once.
    REQUIRE(recorder.ReadCbufValue(1, 8) == env.cbuf_base + 8);
    return recorder.TakeReads();
}

} // namespace

TEST_CASE("Shader reuse matches environments with the same answers", "[shader_reuse]") {
    TestEnvironment source;
    source.native = true;
    const Vulkan::TranslationReads reads{Translate(source)};

    TestEnvironment same;
    same.native = true;
    REQUIRE(reads.Matches(same));
    REQUIRE(Translate(same) == reads);

    TestEnvironment other_value = same;
    other_value.cbuf_base = 4;
    REQUIRE_FALSE(reads.Matches(other_value));

    TestEnvironment not_pointer = same;
    not_pointer.pointer = false;
    REQUIRE_FALSE(reads.Matches(not_pointer));

    TestEnvironment not_indirect_pointer = same;
    not_indirect_pointer.indirect_pointer = false;
    REQUIRE_FALSE(reads.Matches(not_indirect_pointer));

    TestEnvironment not_native = same;
    not_native.native = false;
    REQUIRE_FALSE(reads.Matches(not_native));

    TestEnvironment no_descriptors = same;
    no_descriptors.descriptors = false;
    REQUIRE_FALSE(reads.Matches(no_descriptors));

    TestEnvironment throwing = same;
    throwing.throws = true;
    REQUIRE_FALSE(reads.Matches(throwing));
}

TEST_CASE("Shader reuse cache finds shaders by key and environment", "[shader_reuse]") {
    TestEnvironment source;
    auto shaders = std::make_shared<Vulkan::ReusableShaders>();
    shaders->reads.push_back(Translate(source));

    Vulkan::ShaderTranslationKey key{};
    key.unique_hashes[5] = 0x1234;
    Vulkan::ShaderTranslationKey other_key{key};
    other_key.alpha_test_reference = 1;
    REQUIRE(key == Vulkan::ShaderTranslationKey{key});
    REQUIRE_FALSE(key == other_key);

    Vulkan::ShaderReuseCache cache;
    cache.Add(key, shaders);

    TestEnvironment same;
    std::array<Shader::Environment*, 1> envs{&same};
    REQUIRE(cache.Find(key, envs) == shaders);
    REQUIRE(cache.Find(other_key, envs) == nullptr);

    TestEnvironment different;
    different.pointer = false;
    std::array<Shader::Environment*, 1> different_envs{&different};
    REQUIRE(cache.Find(key, different_envs) == nullptr);

    std::array<Shader::Environment*, 2> too_many{&same, &same};
    REQUIRE(cache.Find(key, too_many) == nullptr);
}

TEST_CASE("Shader recorder preserves descriptor inference and source size", "[shader_reuse]") {
    TestEnvironment env;
    env.descriptors = false;
    Vulkan::RecordingEnvironment recorder{env};
    REQUIRE_FALSE(recorder.CanReadTextureDescriptors());
    REQUIRE(recorder.CodeWords() == env.CodeWords());
    recorder.NoteInferredTextureType(0, Shader::TextureType::Color2D);
    REQUIRE(env.inferred_type == Shader::TextureType::Color2D);
    REQUIRE(recorder.TakeReads().Matches(env));
}

TEST_CASE("Shader reuse cache drops the least recently used keys past its bound",
          "[shader_reuse]") {
    Vulkan::ShaderReuseCache cache;
    TestEnvironment env;
    std::array<Shader::Environment*, 1> envs{&env};
    const auto make_key = [](u64 index) {
        Vulkan::ShaderTranslationKey key{};
        key.unique_hashes[5] = index + 1;
        return key;
    };
    std::vector<std::shared_ptr<Vulkan::ReusableShaders>> added;
    for (u64 index = 0; index < Vulkan::ShaderReuseCache::MaxShaders; ++index) {
        auto shaders = std::make_shared<Vulkan::ReusableShaders>();
        shaders->reads.push_back(Translate(env));
        cache.Add(make_key(index), shaders);
        added.push_back(std::move(shaders));
    }
    // Keys 0 and 1 are used again, so keys 2 onwards are the least recently used.
    REQUIRE(cache.Find(make_key(0), envs) == added[0]);
    REQUIRE(cache.Find(make_key(1), envs) == added[1]);

    auto one_more = std::make_shared<Vulkan::ReusableShaders>();
    one_more->reads.push_back(Translate(env));
    cache.Add(make_key(Vulkan::ShaderReuseCache::MaxShaders), one_more);

    REQUIRE(cache.Find(make_key(0), envs) == added[0]);
    REQUIRE(cache.Find(make_key(1), envs) == added[1]);
    REQUIRE(cache.Find(make_key(Vulkan::ShaderReuseCache::MaxShaders), envs) == one_more);
    REQUIRE(cache.Find(make_key(2), envs) == nullptr);
    // Dropped shaders stay alive for whoever still holds them.
    REQUIRE(added[2]->reads.size() == 1);
}
