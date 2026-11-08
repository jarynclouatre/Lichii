// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

/// Picks the `count` oldest of a set of eviction candidates by last-use frame. Visit each
/// candidate exactly once, in any order, and evict the ones Take() accepts. Ties at the cutoff
/// age are shared out so that no candidate is kept while a newer one is evicted.
class OldestPipelineSelection {
public:
    /// @param last_uses Last-use frames of every candidate. Reordered in place.
    OldestPipelineSelection(std::span<u64> last_uses, size_t count) {
        count = std::min(count, last_uses.size());
        if (count == 0) {
            return;
        }
        std::ranges::nth_element(last_uses, last_uses.begin() + (count - 1));
        cutoff = last_uses[count - 1];
        const size_t older = std::ranges::count_if(
            last_uses, [this](u64 last_use) { return last_use < cutoff; });
        equal_remaining = count - older;
    }

    /// Whether the candidate last used at `last_use` is evicted.
    bool Take(u64 last_use) {
        if (last_use < cutoff) {
            return true;
        }
        if (last_use == cutoff && equal_remaining != 0) {
            --equal_remaining;
            return true;
        }
        return false;
    }

private:
    u64 cutoff{};
    size_t equal_remaining{};
};

/// A pipeline unused for long enough to be evicted.
struct EvictionCandidate {
    u64 last_use;
    /// False for a pipeline prebuilt from the disk cache that no draw has used yet.
    bool used;
};

/// Picks which of `candidates` to evict and returns their indices in ascending order. The first
/// `max_protected` never-used candidates are kept. Of the rest, the `count` oldest are evicted.
/// Protected ones are left out before the oldest are chosen: they are the oldest of all (last use
/// 0), so they would otherwise take the places of evictable ones and nothing would be evicted.
inline std::vector<size_t> SelectPipelinesToEvict(std::span<const EvictionCandidate> candidates,
                                                  size_t max_protected, size_t count) {
    std::vector<size_t> unprotected;
    std::vector<u64> last_uses;
    unprotected.reserve(candidates.size());
    last_uses.reserve(candidates.size());
    size_t protected_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].used && protected_count < max_protected) {
            ++protected_count;
            continue;
        }
        unprotected.push_back(i);
        last_uses.push_back(candidates[i].last_use);
    }
    if (count >= unprotected.size()) {
        return unprotected;
    }
    OldestPipelineSelection oldest{last_uses, count};
    std::vector<size_t> selected;
    selected.reserve(count);
    for (const size_t i : unprotected) {
        if (oldest.Take(candidates[i].last_use)) {
            selected.push_back(i);
        }
    }
    return selected;
}

} // namespace Vulkan
