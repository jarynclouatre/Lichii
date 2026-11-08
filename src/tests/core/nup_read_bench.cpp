// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Random-read benchmark for NUP zstd members. Hidden from normal runs; start it with
//   tests "[nup-bench]"
// NUP_BENCH_FILE=<path.nup> measures a real container (opened read-only, never written); without
// it a synthetic container is packed with the NUP writer under NUP_BENCH_DIR (default: temp).
// NUP_BENCH_SEED, NUP_BENCH_READS, NUP_BENCH_THRASH_READS, NUP_BENCH_CAP_MIB,
// NUP_BENCH_THRASH_CAP_MIB and NUP_BENCH_THREADS override the defaults printed in the report.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/fs/file.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/nup_writer.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using FileSys::NupContainer;

constexpr u64 KiB = 1024;
constexpr u64 MiB = 1024 * KiB;

u64 EnvU64(const char* name, u64 fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
}

// Opens with read access only and lets other processes keep reading and writing the file.
class ReadOnlyFile final : public FileSys::VfsFile {
public:
    explicit ReadOnlyFile(const fs::path& path)
        : file{path, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
               Common::FS::FileShareFlag::ShareReadWrite},
          size{file.IsOpen() ? file.GetSize() : 0}, name{path.filename().string()} {}

    bool IsOpen() const {
        return file.IsOpen();
    }
    std::string GetName() const override {
        return name;
    }
    std::size_t GetSize() const override {
        return size;
    }
    bool Resize(std::size_t) override {
        return false;
    }
    FileSys::VirtualDir GetContainingDirectory() const override {
        return nullptr;
    }
    bool IsWritable() const override {
        return false;
    }
    bool IsReadable() const override {
        return true;
    }
    std::size_t Read(u8* data, std::size_t length, std::size_t offset) const override {
        std::scoped_lock lock{mutex};
        if (!file.Seek(static_cast<s64>(offset))) {
            return 0;
        }
        return file.ReadSpan(std::span{data, length});
    }
    std::size_t Write(const u8*, std::size_t, std::size_t) override {
        return 0;
    }
    bool Rename(std::string_view) override {
        return false;
    }

private:
    mutable std::mutex mutex;
    Common::FS::IOFile file;
    std::size_t size;
    std::string name;
};

// About 3:1 at zstd level 3: literals from a skewed byte distribution mixed with copies of
// recent data, like packed game assets.
std::vector<u8> SyntheticData(u64 size, u64 seed) {
    std::vector<u8> out(size);
    std::mt19937_64 rng{seed};
    std::size_t pos = 0;
    while (pos < out.size()) {
        const u64 r = rng();
        const std::size_t length = std::min<std::size_t>(16 + (r & 255), out.size() - pos);
        if (pos > 4096 && (r >> 8) % 10 < 6) {
            const std::size_t back = 1 + (r >> 16) % std::min<std::size_t>(pos, MiB);
            for (std::size_t i = 0; i < length; ++i) {
                out[pos + i] = out[pos - back + i];
            }
        } else {
            for (std::size_t i = 0; i < length; ++i) {
                const u64 x = rng();
                out[pos + i] = static_cast<u8>(x & (x >> 8));
            }
        }
        pos += length;
    }
    return out;
}

void WriteFile(const fs::path& path, std::span<const u8> data) {
    fs::create_directories(path.parent_path());
    Common::FS::IOFile out{path, Common::FS::FileAccessMode::Write};
    REQUIRE(out.WriteSpan(data) == data.size());
}

struct SyntheticGroup {
    const char* name;
    u64 size;
    int count;
};

constexpr std::array SyntheticGroups{
    SyntheticGroup{"s64k", 64 * KiB, 16},
    SyntheticGroup{"m4m", 4 * MiB, 8},
    SyntheticGroup{"l64m", 64 * MiB, 4},
    SyntheticGroup{"x300m", 300 * MiB, 1},
};

fs::path PackSynthetic(const fs::path& dir) {
    const auto root = dir / "dump";
    const auto archive = dir / "synthetic.nup";
    fs::remove_all(root);
    fs::remove(archive);
    std::string npdm(0x100, '\0');
    npdm.replace(0x80, 4, "ACI0");
    npdm[0x90] = 0x01;
    WriteFile(root / "Base/ExeFS/main", std::vector<u8>(64, 0x11));
    WriteFile(root / "Base/ExeFS/main.npdm",
              std::span{reinterpret_cast<const u8*>(npdm.data()), npdm.size()});
    u64 seed = 1;
    for (const auto& group : SyntheticGroups) {
        for (int i = 0; i < group.count; ++i) {
            WriteFile(root / fmt::format("Base/RomFS/{}_{:02}.bin", group.name, i),
                      SyntheticData(group.size, seed++));
        }
    }
    const FileSys::Nup::PackOptions options{.raw_threshold = 1024 * MiB};
    std::string error;
    REQUIRE(FileSys::Nup::PackDump(root, archive, options, {}, nullptr, &error) ==
            FileSys::Nup::PackResult::Success);
    fs::remove_all(root);
    return archive;
}

struct Member {
    std::size_t index;
    u64 size;
};

struct Summary {
    std::size_t count{};
    double p50{}, p95{}, p99{}, max{}, mean{}, total{};
    std::size_t over_1ms{}, over_16ms{};
};

Summary Summarize(std::vector<double> us) {
    Summary s;
    s.count = us.size();
    if (us.empty()) {
        return s;
    }
    std::sort(us.begin(), us.end());
    const auto at = [&](double q) {
        return us[std::min(us.size() - 1, static_cast<std::size_t>(q * (us.size() - 1) + 0.5))];
    };
    s.p50 = at(0.50);
    s.p95 = at(0.95);
    s.p99 = at(0.99);
    s.max = us.back();
    for (const double v : us) {
        s.total += v;
        s.over_1ms += v > 1000.0;
        s.over_16ms += v > 16000.0;
    }
    s.mean = s.total / static_cast<double>(us.size());
    return s;
}

std::string Us(double us) {
    if (us >= 10000.0) {
        return fmt::format("{:.1f} ms", us / 1000.0);
    }
    if (us >= 1000.0) {
        return fmt::format("{:.2f} ms", us / 1000.0);
    }
    return fmt::format("{:.1f} µs", us);
}

class Bench {
public:
    Bench(FileSys::VirtualFile backing_, fs::path spill_dir_)
        : backing{std::move(backing_)}, spill_dir{std::move(spill_dir_)} {
        NupContainer probe{backing, NupContainer::DefaultCacheCap, spill_dir};
        REQUIRE(probe.GetStatus() == Loader::ResultStatus::Success);
        for (std::size_t i = 0; i < probe.GetFileCount(); ++i) {
            if (probe.IsZstdMember(i)) {
                members.push_back({i, probe.OpenFile(i)->GetSize()});
            }
        }
    }

    const std::vector<Member>& Members() const {
        return members;
    }

    // Latency of each read, in microseconds, against a fresh container (cold decode cache).
    std::vector<double> Random(u64 seed, std::size_t reads, u64 length, bool deep, u64 cap,
                               unsigned threads) const {
        std::vector<Member> eligible;
        for (const auto& m : members) {
            if (m.size >= length) {
                eligible.push_back(m);
            }
        }
        REQUIRE(!eligible.empty());
        std::vector<u64> prefix;
        u64 total = 0;
        for (const auto& m : eligible) {
            prefix.push_back(total += m.size);
        }
        std::mt19937_64 rng{seed};
        std::vector<ReadPlan> plan(reads);
        for (auto& read : plan) {
            const u64 pick = rng() % total;
            read.member = static_cast<std::size_t>(
                std::upper_bound(prefix.begin(), prefix.end(), pick) - prefix.begin());
            const u64 size = eligible[read.member].size;
            const u64 first = deep ? size - std::max(length, size / 4) : 0;
            read.offset = first + rng() % (size - length - first + 1);
        }

        return Run(eligible, plan, length, cap, threads);
    }

    // Streams every member from its start in `length` chunks, all members interleaved, as a game
    // playing several audio or movie streams at once would.
    std::vector<double> Sequential(std::size_t reads, u64 length, u64 cap) const {
        std::vector<ReadPlan> plan;
        for (u64 offset = 0; plan.size() < reads; offset += length) {
            const std::size_t before = plan.size();
            for (std::size_t i = 0; i < members.size() && plan.size() < reads; ++i) {
                if (offset + length <= members[i].size) {
                    plan.push_back({i, offset});
                }
            }
            if (plan.size() == before) {
                break;
            }
        }
        return Run(members, plan, length, cap, 1);
    }

    struct FirstTouch {
        std::vector<double> at_start, at_deep;
        double full_mib_s{};
    };

    // First 4 KiB read of each member at offset 0 and at 90 %, and a sequential full read.
    FirstTouch Touch(const std::vector<Member>& picked) const {
        FirstTouch out;
        std::vector<u8> buffer(MiB);
        {
            NupContainer nup{backing, NupContainer::DefaultCacheCap, spill_dir};
            for (const auto& m : picked) {
                const auto file = nup.OpenFile(m.index);
                const auto start = Clock::now();
                REQUIRE(file->Read(buffer.data(), 4 * KiB, 0) == std::min<u64>(4 * KiB, m.size));
                out.at_start.push_back(
                    std::chrono::duration<double, std::micro>(Clock::now() - start).count());
            }
        }
        {
            NupContainer nup{backing, NupContainer::DefaultCacheCap, spill_dir};
            for (const auto& m : picked) {
                const auto file = nup.OpenFile(m.index);
                const u64 offset = (m.size - std::min<u64>(4 * KiB, m.size)) * 9 / 10;
                const auto start = Clock::now();
                REQUIRE(file->Read(buffer.data(), 4 * KiB, offset) ==
                        std::min<u64>(4 * KiB, m.size - offset));
                out.at_deep.push_back(
                    std::chrono::duration<double, std::micro>(Clock::now() - start).count());
            }
        }
        {
            NupContainer nup{backing, NupContainer::DefaultCacheCap, spill_dir};
            u64 bytes = 0;
            const auto start = Clock::now();
            for (const auto& m : picked) {
                const auto file = nup.OpenFile(m.index);
                for (u64 offset = 0; offset < m.size; offset += buffer.size()) {
                    const u64 want = std::min<u64>(buffer.size(), m.size - offset);
                    REQUIRE(file->Read(buffer.data(), want, offset) == want);
                }
                bytes += m.size;
            }
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            out.full_mib_s = static_cast<double>(bytes) / static_cast<double>(MiB) / seconds;
        }
        return out;
    }

private:
    struct ReadPlan {
        std::size_t member;
        u64 offset;
    };

    std::vector<double> Run(const std::vector<Member>& eligible, const std::vector<ReadPlan>& plan,
                            u64 length, u64 cap, unsigned threads) const {
        const std::size_t reads = plan.size();
        NupContainer nup{backing, cap, spill_dir};
        std::vector<FileSys::VirtualFile> files(eligible.size());
        for (std::size_t i = 0; i < eligible.size(); ++i) {
            files[i] = nup.OpenFile(eligible[i].index);
        }
        std::vector<double> us(reads);
        std::atomic<std::size_t> short_reads{};
        const auto run = [&](unsigned worker) {
            std::vector<u8> buffer(length);
            for (std::size_t i = worker; i < plan.size(); i += threads) {
                const auto start = Clock::now();
                const auto got = files[plan[i].member]->Read(buffer.data(), length, plan[i].offset);
                us[i] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                short_reads += got != length;
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < threads; ++t) {
            pool.emplace_back(run, t);
        }
        run(0);
        for (auto& t : pool) {
            t.join();
        }
        REQUIRE(short_reads == 0);
        return us;
    }

    FileSys::VirtualFile backing;
    fs::path spill_dir;
    std::vector<Member> members;
};

} // namespace

TEST_CASE("NUP random read benchmark", "[.][benchmark][nup-bench]") {
    const u64 seed = EnvU64("NUP_BENCH_SEED", 0x4E5550);
    const std::size_t reads = EnvU64("NUP_BENCH_READS", 1000);
    const std::size_t thrash_reads = EnvU64("NUP_BENCH_THRASH_READS", 300);
    const u64 cap = EnvU64("NUP_BENCH_CAP_MIB", NupContainer::DefaultCacheCap / MiB) * MiB;
    const unsigned threads = static_cast<unsigned>(EnvU64("NUP_BENCH_THREADS", 4));

    const char* real = std::getenv("NUP_BENCH_FILE");
    const fs::path work = [] {
        const char* dir = std::getenv("NUP_BENCH_DIR");
        return (dir != nullptr && *dir != '\0' ? fs::path{dir} : fs::temp_directory_path()) /
               "nup-bench";
    }();
    fs::create_directories(work);
    const auto spill_dir = work / "spill";

    fs::path archive;
    if (real != nullptr && *real != '\0') {
        archive = real;
    } else {
        const auto start = Clock::now();
        archive = PackSynthetic(work);
        fmt::print("packed {} in {:.1f} s\n", archive.string(),
                   std::chrono::duration<double>(Clock::now() - start).count());
    }
    const auto backing = std::make_shared<ReadOnlyFile>(archive);
    REQUIRE(backing->IsOpen());
    Bench bench{backing, spill_dir};
    const u64 thrash_cap =
        EnvU64("NUP_BENCH_THRASH_CAP_MIB", real != nullptr && *real != '\0' ? 128 : 160) * MiB;

    u64 zstd_bytes = 0;
    for (const auto& m : bench.Members()) {
        zstd_bytes += m.size;
    }
    fmt::print("\n{}: {:.2f} GiB, {} zstd members holding {:.2f} GiB; seed {:#x}, cache cap {} "
               "MiB, thrash cap {} MiB, {} threads\n",
               archive.filename().string(),
               static_cast<double>(backing->GetSize()) / static_cast<double>(1024 * MiB),
               bench.Members().size(), static_cast<double>(zstd_bytes) / (1024.0 * MiB), seed,
               cap / MiB, thrash_cap / MiB, threads);

    struct Bucket {
        const char* label;
        u64 below;
    };
    constexpr std::array Buckets{Bucket{"≤ 256 KiB", 256 * KiB + 1}, Bucket{"≤ 8 MiB", 8 * MiB + 1},
                                 Bucket{"≤ 128 MiB", 128 * MiB + 1}, Bucket{"> 128 MiB", ~u64{0}}};
    fmt::print("\n| member size | n | 4 KiB @ 0: p50 | max | 4 KiB @ 90 %: p50 | max | full read "
               "|\n|---|---:|---:|---:|---:|---:|---:|\n");
    u64 above = 0;
    std::mt19937_64 rng{seed};
    for (const auto& bucket : Buckets) {
        std::vector<Member> picked;
        for (const auto& m : bench.Members()) {
            if (m.size > above && m.size < bucket.below) {
                picked.push_back(m);
            }
        }
        above = bucket.below - 1;
        if (picked.empty()) {
            continue;
        }
        std::shuffle(picked.begin(), picked.end(), rng);
        picked.resize(std::min<std::size_t>(picked.size(), 8));
        const auto touch = bench.Touch(picked);
        const auto start = Summarize(touch.at_start);
        const auto deep = Summarize(touch.at_deep);
        fmt::print("| {} | {} | {} | {} | {} | {} | {:.0f} MiB/s |\n", bucket.label, picked.size(),
                   Us(start.p50), Us(start.max), Us(deep.p50), Us(deep.max), touch.full_mib_s);
    }

    fmt::print("\n| random reads | cap | n | p50 | p95 | p99 | max | mean | > 1 ms | > 16 ms | "
               "sum |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    const auto row = [&](const char* label, u64 row_cap, const std::vector<double>& us) {
        const auto s = Summarize(us);
        fmt::print("| {} | {} MiB | {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} s |\n", label,
                   row_cap / MiB, s.count, Us(s.p50), Us(s.p95), Us(s.p99), Us(s.max), Us(s.mean),
                   s.over_1ms, s.over_16ms, s.total / 1e6);
    };
    row("4 KiB uniform", cap, bench.Random(seed, reads, 4 * KiB, false, cap, 1));
    row("64 KiB uniform", cap, bench.Random(seed + 1, reads, 64 * KiB, false, cap, 1));
    row("4 KiB deep (last 25 %)", cap, bench.Random(seed + 2, reads, 4 * KiB, true, cap, 1));
    row("4 KiB uniform, threads", cap, bench.Random(seed, reads, 4 * KiB, false, cap, threads));
    row("64 KiB sequential streams", cap, bench.Sequential(reads, 64 * KiB, cap));
    row("4 KiB uniform, thrash", thrash_cap,
        bench.Random(seed + 3, thrash_reads, 4 * KiB, false, thrash_cap, 1));
    std::fflush(stdout);
    std::error_code ec;
    fs::remove_all(spill_dir, ec);
}
