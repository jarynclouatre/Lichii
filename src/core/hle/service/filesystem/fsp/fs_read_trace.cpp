// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <vector>

#include "common/logging.h"

namespace Service::FileSystem {
namespace {

using Clock = std::chrono::steady_clock;

struct ReadStats {
    const char* label;
    std::mutex mutex{};
    Clock::time_point window_start{};
    std::vector<Clock::duration> latencies{};
    u64 window_bytes{};
    u64 total_reads{};
    u64 total_bytes{};
    Clock::duration total_time{};
};

std::array<ReadStats, 2> stats{{{"storage"}, {"file"}}};

double Micros(Clock::duration duration) {
    return std::chrono::duration<double, std::micro>(duration).count();
}

double MiB(u64 bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

void RecordRead(ReadTraceSource source, u64 bytes, Clock::duration elapsed) {
    auto& s = stats[static_cast<std::size_t>(source)];
    std::scoped_lock lock{s.mutex};
    const auto now = Clock::now();
    if (s.latencies.empty()) {
        s.window_start = now - elapsed;
    }
    s.latencies.push_back(elapsed);
    s.window_bytes += bytes;
    ++s.total_reads;
    s.total_bytes += bytes;
    s.total_time += elapsed;
    if (now - s.window_start < std::chrono::seconds{1}) {
        return;
    }

    auto& l = s.latencies;
    std::sort(l.begin(), l.end());
    const auto at = [&](double q) { return l[static_cast<std::size_t>(q * (l.size() - 1))]; };
    const auto over = [&](auto limit) {
        return l.end() - std::upper_bound(l.begin(), l.end(), Clock::duration{limit});
    };
    LOG_INFO(Service_FS,
             "Guest {} reads: {} in {:.2f} s, {:.2f} MiB, p50 {:.1f} us, p99 {:.1f} us, max {:.1f} "
             "us, {} over 1 ms, {} over 16 ms; since boot {} reads, {:.1f} MiB, {:.3f} s reading",
             s.label, l.size(), std::chrono::duration<double>(now - s.window_start).count(),
             MiB(s.window_bytes), Micros(at(0.50)), Micros(at(0.99)), Micros(l.back()),
             over(std::chrono::milliseconds{1}), over(std::chrono::milliseconds{16}), s.total_reads,
             MiB(s.total_bytes), std::chrono::duration<double>(s.total_time).count());
    l.clear();
    s.window_bytes = 0;
}

} // namespace Service::FileSystem
