// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "common/thread_worker.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"

namespace Vulkan {

/// EDEN_NX2_FAST_FIRST=1: a pipeline built while the game runs is first created with
/// VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT and drawn with as soon as the driver returns it;
/// the optimized build follows on a worker and takes its place at a later draw. Trades the seconds
/// the driver spends on the heaviest pipelines (a demanding Switch 2 test title, first visit:
/// draws held up to 7 s of every 10 s) for a slower shader during that time.
inline bool FastFirstBuildEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FAST_FIRST");
        return value && *value && *value != '0';
    }();
    return enabled;
}

/// EDEN_NX2_HEAVY_SKIP: a native pipeline of at least this many converted instructions (all
/// stages) has its draws skipped while the driver builds it, instead of holding every draw
/// behind it. "1" picks 4000; 0 or unset turns it off. In a demanding Switch 2 test title's first
/// visits every driver build over 500 ms was above 4000, and they made ~90% of the compile time.
inline size_t HeavySkipWords() {
    static const size_t words = [] {
        const char* value = std::getenv("EDEN_NX2_HEAVY_SKIP");
        if (!value || !*value) {
            return size_t{0};
        }
        const size_t parsed = static_cast<size_t>(std::strtoull(value, nullptr, 10));
        return parsed == 1 ? size_t{4000} : parsed;
    }();
    return words;
}

class EmissionBudget {
public:
    class Reservation {
    public:
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        ~Reservation() {
            if (pending) {
                pending->fetch_sub(1, std::memory_order_relaxed);
            }
        }

    private:
        friend class EmissionBudget;
        Reservation() = default;
        std::shared_ptr<std::atomic<size_t>> pending;
    };

    explicit EmissionBudget(size_t capacity_) : capacity{capacity_} {}

    std::unique_ptr<Reservation> TryAcquire(bool native) {
        size_t count = pending->load(std::memory_order_relaxed);
        if (!DefersShaderEmission(native, count, capacity)) {
            return nullptr;
        }
        auto reservation = std::unique_ptr<Reservation>{new Reservation};
        while (DefersShaderEmission(native, count, capacity)) {
            if (pending->compare_exchange_weak(count, count + 1, std::memory_order_relaxed)) {
                reservation->pending = pending;
                return reservation;
            }
        }
        return nullptr;
    }

    size_t Pending() const {
        return pending->load(std::memory_order_relaxed);
    }

private:
    const size_t capacity;
    std::shared_ptr<std::atomic<size_t>> pending = std::make_shared<std::atomic<size_t>>(0);
};

class PipelineBuild {
    struct State {
        explicit State(std::function<void()> task_) : task{std::move(task_)} {}

        void Run() noexcept {
            if (claimed.exchange(true)) {
                return;
            }
            try {
                task();
            } catch (const std::exception& exception) {
                failed = true;
                LOG_ERROR(Render_Vulkan, "Graphics pipeline worker failed: {}", exception.what());
            } catch (...) {
                failed = true;
                LOG_ERROR(Render_Vulkan,
                          "Graphics pipeline worker failed with an unknown exception");
            }
            task = {};
            std::scoped_lock lock{mutex};
            complete.store(true, std::memory_order_release);
            condition.notify_all();
        }

        std::function<void()> task;
        bool failed{};
        std::atomic_bool claimed{};
        std::atomic_bool complete{};
        std::mutex mutex;
        std::condition_variable condition;
    };

public:
    PipelineBuild(Common::ThreadWorker* workers, std::function<void()> task)
        : state{std::make_shared<State>(std::move(task))} {
        if (workers) {
            workers->QueueWork([state = state] { state->Run(); });
        } else {
            state->Run();
        }
    }

    ~PipelineBuild() {
        Wait(true);
    }

    bool IsBuilt() const {
        return state->complete.load(std::memory_order_acquire);
    }

    bool Failed() const {
        return IsBuilt() && state->failed;
    }

    void Wait(bool may_build = false) const {
        const auto shared = state;
        if (may_build) {
            shared->Run();
        }
        std::unique_lock lock{shared->mutex};
        shared->condition.wait(lock,
                               [&] { return shared->complete.load(std::memory_order_acquire); });
    }

    /// Waits up to `timeout` for the build, without running it on this thread. The worker sets
    /// `complete` under the mutex before notifying, so the wakeup cannot be lost.
    /// @return True when the pipeline is built.
    bool WaitFor(std::chrono::microseconds timeout) const {
        const auto shared = state;
        std::unique_lock lock{shared->mutex};
        return shared->condition.wait_for(lock, timeout, [&] {
            return shared->complete.load(std::memory_order_acquire);
        });
    }

    PipelineBuild(const PipelineBuild&) = delete;
    PipelineBuild& operator=(const PipelineBuild&) = delete;

private:
    std::shared_ptr<State> state;
};

} // namespace Vulkan
