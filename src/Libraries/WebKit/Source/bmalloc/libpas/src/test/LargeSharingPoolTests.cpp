/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "TestHarness.h"

#if TLC

#include "LargeSharingPoolDump.h"
#include "pas_epoch.h"
#include "pas_heap_lock.h"
#include "pas_large_sharing_pool.h"
#include "pas_page_malloc.h"
#include "pas_physical_memory_transaction.h"
#include "pas_scavenger.h"
#include <vector>

#define PG (pas_page_malloc_alignment())
#define END ((uintptr_t)1 << PAS_ADDRESS_BITS)

using namespace std;

namespace {

struct Range {
    Range(uintptr_t begin,
          uintptr_t end,
          pas_commit_mode isCommitted,
          size_t numLiveBytes,
          uint64_t epoch)
        : begin(begin)
        , end(end)
        , isCommitted(isCommitted)
        , numLiveBytes(numLiveBytes)
        , epoch(epoch)
    {
    }

    explicit Range(pas_large_sharing_node* node)
        : begin(node->range.begin)
        , end(node->range.end)
        , isCommitted(node->is_committed)
        , numLiveBytes(node->num_live_bytes)
        , epoch(node->use_epoch)
    {
    }

    bool operator==(Range other) const
    {
        return begin == other.begin
            && end == other.end
            && isCommitted == other.isCommitted
            && numLiveBytes == other.numLiveBytes
            && epoch == other.epoch;
    }

    bool operator!=(Range other) const { return !(*this == other); }
    
    uintptr_t begin;
    uintptr_t end;
    pas_commit_mode isCommitted;
    size_t numLiveBytes;
    uint64_t epoch;
};

ostream& operator<<(ostream& out, Range range)
{
    out << reinterpret_cast<void*>(range.begin) << "..." << reinterpret_cast<void*>(range.end)
        << ", " << (range.isCommitted ? "committed" : "decommitted") << ", "
        << range.numLiveBytes << "/" << (range.end - range.begin) << ", " << range.epoch;
    return out;
}

void assertState(const vector<Range>& ranges)
{
    vector<pas_large_sharing_node*> nodes = largeSharingPoolAsVector();
    
    bool allGood = true;

    if (nodes.size() != ranges.size()) {
        cout << "State does not match because we expected " << ranges.size() << " ranges but got "
             << nodes.size() << " ranges.\n";
        allGood = false;
    } else {
        for (size_t index = 0; index < nodes.size(); ++index) {
            pas_large_sharing_node* node = nodes[index];
            Range actualRange(node);
            Range expectedRange = ranges[index];
            if (expectedRange != actualRange) {
                cout << "State does not match at index " << index << ": expected:\n"
                     << "    " << expectedRange << ", but got:\n"
                     << "    " << actualRange << "\n";
                allGood = false;
            }
        }
    }

    if (!allGood) {
        cout << "Got mismatch in states. Expected the state to be:\n";
        for (Range range : ranges)
            cout << "    " << range << "\n";
        cout << "But got:\n";
        for (pas_large_sharing_node* node : nodes)
            cout << "    " << Range(node) << "\n";
    }

    CHECK(allGood);
}

void testGoodCoalesceEpochUpdate()
{
    static constexpr bool verbose = false;
    
    pas_physical_memory_transaction transaction;
    
    pas_scavenger_suspend();
    pas_physical_memory_transaction_construct(&transaction);

    CHECK_EQUAL(pas_current_epoch, 0);
    
    pas_heap_lock_lock();
    pas_large_sharing_pool_boot_free(
        pas_range_create(10 * PG, 20 * PG),
        pas_physical_memory_is_locked_by_virtual_range_common_lock,
        pas_may_mmap);
    pas_heap_lock_unlock();

    assertState({ Range(0, 10 * PG, pas_committed, 10 * PG, 0),
                  Range(10 * PG, 20 * PG, pas_committed, 0, 1),
                  Range(20 * PG, END, pas_committed, END - 20 * PG, 0) });
    
    pas_heap_lock_lock();
    pas_large_sharing_pool_boot_free(
        pas_range_create(20 * PG, 30 * PG),
        pas_physical_memory_is_locked_by_virtual_range_common_lock,
        pas_may_mmap);
    pas_heap_lock_unlock();

    assertState({ Range(0, 10 * PG, pas_committed, 10 * PG, 0),
                  Range(10 * PG, 20 * PG, pas_committed, 0, 1),
                  Range(20 * PG, 30 * PG, pas_committed, 0, 2),
                  Range(30 * PG, END, pas_committed, END - 30 * PG, 0) });
    
    pas_heap_lock_lock();
    CHECK(pas_large_sharing_pool_allocate_and_commit(
              pas_range_create(10 * PG, 30 * PG),
              &transaction,
              pas_physical_memory_is_locked_by_virtual_range_common_lock,
              pas_may_mmap));
    pas_heap_lock_unlock();
    
    if (verbose)
        dumpLargeSharingPool();

    assertState({ Range(0, END, pas_committed, END, 3) });
}

} // anonymous namespace

#endif // TLC

void addLargeSharingPoolTests()
{
#if TLC
    EpochIsCounter epochIsCounter;
    
    ADD_TEST(testGoodCoalesceEpochUpdate());
#endif // TLC
}

