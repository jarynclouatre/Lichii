// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace {
namespace IR = Shader::IR;
using IR::Opcode;

constexpr f32 Inf = std::numeric_limits<f32>::infinity();
constexpr f32 NaN = std::numeric_limits<f32>::quiet_NaN();

class FmzEnvironment final : public Shader::Environment {
public:
    u64 ReadInstruction(u32) override { throw std::logic_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32, u32) override { throw std::logic_error("Unexpected cbuf read"); }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

f64 HalfValue(u16 bits) {
    const int exponent = (bits >> 10) & 31;
    const f64 fraction = bits & 1023;
    const f64 magnitude = exponent == 31
                              ? (fraction == 0 ? std::numeric_limits<f64>::infinity()
                                               : std::numeric_limits<f64>::quiet_NaN())
                              : std::ldexp(exponent == 0 ? fraction : fraction + 1024,
                                           exponent == 0 ? -24 : exponent - 25);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

// The test values are exact in binary16, so no rounding is needed.
u16 HalfBits(f64 value) {
    for (u32 bits = 0; bits < 65536; ++bits) {
        const f64 half = HalfValue(static_cast<u16>(bits));
        if (half == value || (std::isnan(value) && std::isnan(half))) {
            return static_cast<u16>(bits);
        }
    }
    throw std::logic_error("Value is not exact in binary16");
}

u32 Halves(f64 low, f64 high) {
    return HalfBits(low) | u32{HalfBits(high)} << 16;
}

f64 Saturate(f64 value) {
    return std::isnan(value) ? 0.0 : std::clamp(value, 0.0, 1.0);
}

// Translates one instruction writing R0 from R2 (a), R3 (b) and R4 (c), and evaluates the IR with
// the GPU's results: FPSaturate turns NaN into 0, as the hardware's .SAT does.
struct FmzProgram {
    Shader::ObjectPool<IR::Inst> pool{64};
    IR::Block block{pool};
    FmzEnvironment env;
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    std::unordered_map<const IR::Inst*, u32> values;
    std::array<u32, IR::NUM_REGS> registers{};

    u32 Value(const IR::Value& value) const {
        if (!value.IsImmediate()) {
            return values.at(value.InstRecursive());
        }
        switch (value.Type()) {
        case IR::Type::U1:
            return value.U1();
        case IR::Type::U32:
            return value.U32();
        case IR::Type::F32:
            return std::bit_cast<u32>(value.F32());
        default:
            throw std::logic_error("Unsupported immediate");
        }
    }

    u32 Evaluate(const IR::Inst& inst) const {
        const auto a = [&](size_t index) { return Value(inst.Arg(index)); };
        const auto f = [&](size_t index) -> f64 { return std::bit_cast<f32>(a(index)); };
        const auto h = [&](size_t index) { return HalfValue(static_cast<u16>(a(index))); };
        const auto f32_bits = [](f64 value) { return std::bit_cast<u32>(static_cast<f32>(value)); };
        switch (inst.GetOpcode()) {
        case Opcode::Identity:
        case Opcode::BitCastF32U32:
        case Opcode::BitCastU32F32:
        case Opcode::UnpackFloat2x16:
        case Opcode::PackFloat2x16:
            return a(0);
        case Opcode::GetRegister:
            return registers.at(static_cast<size_t>(inst.Arg(0).Reg()));
        case Opcode::CompositeExtractF16x2:
            return (a(0) >> (a(1) * 16)) & 0xffff;
        case Opcode::CompositeConstructF16x2:
            return (a(0) & 0xffff) | a(1) << 16;
        case Opcode::ConvertF32F16:
            return f32_bits(h(0));
        case Opcode::ConvertF16F32:
            return HalfBits(f(0));
        case Opcode::FPOrdEqual32:
            return f(0) == f(1);
        case Opcode::LogicalOr:
            return a(0) != 0 || a(1) != 0;
        case Opcode::SelectF32:
            return a(0) != 0 ? a(1) : a(2);
        case Opcode::FPMul32:
            return f32_bits(static_cast<f32>(f(0)) * static_cast<f32>(f(1)));
        case Opcode::FPFma32:
            return f32_bits(std::fma(static_cast<f32>(f(0)), static_cast<f32>(f(1)),
                                     static_cast<f32>(f(2))));
        case Opcode::FPSaturate32:
            return f32_bits(Saturate(f(0)));
        case Opcode::FPMul16:
            return HalfBits(h(0) * h(1));
        case Opcode::FPFma16:
            return HalfBits(std::fma(h(0), h(1), h(2)));
        case Opcode::FPSaturate16:
            return HalfBits(Saturate(h(0)));
        default:
            throw std::logic_error(std::string{IR::NameOf(inst.GetOpcode())});
        }
    }

    u32 Run(u32 a, u32 b, u32 c) {
        registers.fill(0xdeadbeefU);
        registers[2] = a;
        registers[3] = b;
        registers[4] = c;
        values.clear();
        for (const IR::Inst& inst : block.Instructions()) {
            if (inst.GetOpcode() == Opcode::SetRegister) {
                registers.at(static_cast<size_t>(inst.Arg(0).Reg())) = Value(inst.Arg(1));
            } else if (inst.GetOpcode() != Opcode::Void) {
                values.emplace(&inst, Evaluate(inst));
            }
        }
        return registers[0];
    }
};

// Register form: R0 = R2 * R3 + R4, FMZ in bits 53-54, SAT in bit 50.
u64 FfmaReg(bool sat) {
    return u64{2} << 8 | u64{3} << 20 | u64{4} << 39 | u64{sat} << 50 | u64{2} << 53;
}

// Register form with H1_H0 swizzles and merge: FMZ in bits 37-38, SAT in bit 32.
u64 Hfma2Reg(bool sat, bool f32_addend = false) {
    return u64{2} << 8 | u64{3} << 20 | u64{sat} << 32 | u64{f32_addend} << 35 | u64{2} << 37 |
           u64{4} << 39;
}

// Register form with H1_H0 swizzles and merge: FMZ in bits 39-40, SAT in bit 32.
u64 Hmul2Reg(bool sat) {
    return u64{2} << 8 | u64{3} << 20 | u64{sat} << 32 | u64{2} << 39;
}

u32 Bits(f32 value) {
    return std::bit_cast<u32>(value);
}
} // Anonymous namespace

TEST_CASE("Maxwell FFMA.FMZ keeps the addend when a factor is zero, with or without SAT",
          "[shader][maxwell][fmz]") {
    struct Case {
        f32 a, b, c;
    };
    const std::array<Case, 7> cases{{
        {0.0f, Inf, 0.75f},
        {Inf, 0.0f, 0.75f},
        {NaN, 0.0f, 1.5f},
        {-0.0f, NaN, -2.0f},
        {2.0f, 3.0f, 0.25f},
        {2.0f, 0.25f, 0.25f},
        {Inf, 2.0f, 1.0f},
    }};
    for (const bool sat : {false, true}) {
        FmzProgram program;
        program.visitor.FFMA_reg(FfmaReg(sat));
        for (const Case& test : cases) {
            const bool zero = test.a == 0.0f || test.b == 0.0f;
            const f64 value = zero ? test.c : std::fma(test.a, test.b, test.c);
            const f32 expected = static_cast<f32>(sat ? Saturate(value) : value);
            INFO("a " << test.a << " b " << test.b << " c " << test.c << " sat " << sat);
            REQUIRE(program.Run(Bits(test.a), Bits(test.b), Bits(test.c)) == Bits(expected));
        }
    }
}

TEST_CASE("Maxwell HFMA2.FMZ translates half operands and keeps the addend",
          "[shader][maxwell][fmz]") {
    // Each case is two lanes: {a, b, c}.
    struct Case {
        std::array<f64, 3> low, high;
    };
    const std::array<Case, 3> cases{{
        {{0.0, Inf, 0.75}, {2.0, 0.25, 0.25}},
        {{NaN, 0.0, 0.5}, {0.0, 0.0, 2.0}},
        {{-0.5, Inf, 1.0}, {0.0, NaN, -1.5}},
    }};
    for (const bool sat : {false, true}) {
        FmzProgram program;
        program.visitor.HFMA2_reg(Hfma2Reg(sat));
        for (const Case& test : cases) {
            const auto lane = [&](const std::array<f64, 3>& op) {
                const bool zero = op[0] == 0.0 || op[1] == 0.0;
                const f64 value = zero ? op[2] : std::fma(op[0], op[1], op[2]);
                return sat ? Saturate(value) : value;
            };
            INFO("sat " << sat << " low c " << test.low[2] << " high c " << test.high[2]);
            const u32 result = program.Run(Halves(test.low[0], test.high[0]),
                                           Halves(test.low[1], test.high[1]),
                                           Halves(test.low[2], test.high[2]));
            REQUIRE(result == Halves(lane(test.low), lane(test.high)));
        }
    }
}

TEST_CASE("Maxwell HFMA2.FMZ with an F32 addend keeps the addend", "[shader][maxwell][fmz]") {
    for (const bool sat : {false, true}) {
        FmzProgram program;
        program.visitor.HFMA2_reg(Hfma2Reg(sat, true));
        const u32 result = program.Run(Halves(0.0, 1.0), Halves(Inf, 0.5), Bits(0.75f));
        REQUIRE(result == (sat ? Halves(0.75, 1.0) : Halves(0.75, 1.25)));
    }
}

TEST_CASE("Maxwell HMUL2.FMZ translates half operands and gives zero for a zero factor",
          "[shader][maxwell][fmz]") {
    for (const bool sat : {false, true}) {
        FmzProgram program;
        program.visitor.HMUL2_reg(Hmul2Reg(sat));
        REQUIRE(program.Run(Halves(0.0, 1.5), Halves(Inf, 2.0), 0) ==
                (sat ? Halves(0.0, 1.0) : Halves(0.0, 3.0)));
        REQUIRE(program.Run(Halves(NaN, -0.5), Halves(0.0, 0.5), 0) ==
                Halves(0.0, sat ? 0.0 : -0.25));
    }
}
