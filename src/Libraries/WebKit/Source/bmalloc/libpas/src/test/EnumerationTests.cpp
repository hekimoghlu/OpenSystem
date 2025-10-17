/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include "pas_root.h"
#include "bmalloc_heap.h"
#include "pas_probabilistic_guard_malloc_allocator.h"
#include "pas_heap.h"
#include "iso_heap.h"
#include "iso_heap_config.h"
#include "pas_heap_ref_kind.h"

#include <map>
#include <stdlib.h>
#include <set>
#include <sys/mman.h>

using namespace std;

namespace {

const bool verbose = false;

struct PageRange {
    PageRange() = default;

    PageRange(void* base, size_t size)
        : base(base)
        , size(size)
    {
        PAS_ASSERT(pas_is_aligned(reinterpret_cast<uintptr_t>(base), pas_page_malloc_alignment()));
        PAS_ASSERT(pas_is_aligned(size, pas_page_malloc_alignment()));
        PAS_ASSERT(!!base == !!size);
    }

    bool operator<(PageRange range) const
    {
        return base < range.base;
    }

    void* end() const
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + size);
    }

    void* base { nullptr };
    size_t size { 0 };
};

set<PageRange> pageRanges;

struct RecordedRange {
    RecordedRange() = default;

    RecordedRange(void* base, size_t size)
        : base(base)
        , size(size)
    {
        PAS_ASSERT(base);
    }

    bool operator<(RecordedRange other) const
    {
        return base < other.base;
    }

    void* end() const
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + size);
    }

    void* base { nullptr };
    size_t size { 0 };
};

struct ReaderRange {
    ReaderRange() = default;

    ReaderRange(void* base, size_t size)
        : base(base)
        , size(size)
    {
        PAS_ASSERT(base);
        PAS_ASSERT(size);
    }

    bool operator<(ReaderRange other) const
    {
        if (base != other.base)
            return base < other.base;
        return size < other.size;
    }

    void* base { nullptr };
    size_t size { 0 };
};

map<pas_enumerator_record_kind, set<RecordedRange>> recordedRanges;
map<ReaderRange, void*> readerCache;

void* enumeratorReader(pas_enumerator* enumerator, void* address, size_t size, void* arg)
{
    CHECK(!arg);
    CHECK(size);

    ReaderRange range = ReaderRange(address, size);

    auto readerCacheIter = readerCache.find(range);
    if (readerCacheIter != readerCache.end())
        return readerCacheIter->second;

    void* result = pas_enumerator_allocate(enumerator, size);

    void* pageAddress = reinterpret_cast<void*>(
        pas_round_down_to_power_of_2(
            reinterpret_cast<uintptr_t>(address),
            pas_page_malloc_alignment()));
    size_t pagesSize =
        pas_round_up_to_power_of_2(
            reinterpret_cast<uintptr_t>(address) + size,
            pas_page_malloc_alignment())
        - reinterpret_cast<uintptr_t>(pageAddress);
    void* pageEndAddress = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(pageAddress) + pagesSize);

    bool areProtectedPages = false;
    if (!pageRanges.empty()) {
        auto pageRangeIter = pageRanges.upper_bound(PageRange(pageAddress, pagesSize));
        if (pageRangeIter != pageRanges.begin()) {
            --pageRangeIter;
            areProtectedPages =
                pageRangeIter->base <= pageAddress
                && pageRangeIter->end() > pageAddress;
        }
    }

    if (verbose) {
        cout << "address = " << address << "..."
             << reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) + size)
             << ", pageAddress = " << pageAddress << "..." << pageEndAddress
             << ", areProtectedPages = " << areProtectedPages << "\n";
    }

    if (areProtectedPages) {
        int systemResult = mprotect(pageAddress, pagesSize, PROT_READ);
        PAS_ASSERT(!systemResult);
    }

    memcpy(result, address, size);

    if (areProtectedPages) {
        int systemResult = mprotect(pageAddress, pagesSize, PROT_NONE);
        PAS_ASSERT(!systemResult);
    }

    readerCache[range] = result;
    return result;
}

void enumeratorRecorder(pas_enumerator* enumerator, void* address, size_t size, pas_enumerator_record_kind kind, void* arg)
{
    PAS_UNUSED_PARAM(enumerator);
    CHECK(size);
    CHECK(!arg);

    if (verbose) {
        cout << "Recording " << pas_enumerator_record_kind_get_string(kind) << ":" << address
             << "..." << reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) + size) << "\n";
    }

    RecordedRange range = RecordedRange(address, size);

    CHECK(!recordedRanges[kind].count(range));
    recordedRanges[kind].insert(range);
}


void testBasicEnumeration() {

    pas_heap_lock_lock();
    pas_root* root = pas_root_create();
    pas_heap_lock_unlock();

    auto size = 25;
    void* arr[size];
    for (auto i = 0; i < size; i++) {
        arr[i] = bmalloc_try_allocate(1000000, pas_non_compact_allocation_mode);
        PAS_ASSERT(arr[i]);
    }

    pas_enumerator* enumerator = pas_enumerator_create(root, enumeratorReader, nullptr, enumeratorRecorder, nullptr, pas_enumerator_record_meta_records, pas_enumerator_record_payload_records, pas_enumerator_record_object_records);
    pas_enumerator_enumerate_all(enumerator);

    pas_enumerator_destroy(enumerator);
}

void testPGMEnumerationBasic() {

    pas_heap_lock_lock();
    pas_root* root = pas_root_create();
    pas_heap_lock_unlock();

    pas_heap_ref heapRef = ISO_HEAP_REF_INITIALIZER_WITH_ALIGNMENT(getpagesize() * 100, getpagesize());
    pas_heap* heap = iso_heap_ref_get_heap(&heapRef);
    pas_physical_memory_transaction transaction;
    pas_physical_memory_transaction_construct(&transaction);

    pas_heap_lock_lock();

    size_t alloc_size = 16384;
    pas_allocation_result result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    pas_heap_lock_unlock();

    pas_enumerator* enumerator = pas_enumerator_create(root, enumeratorReader, nullptr, enumeratorRecorder, nullptr, pas_enumerator_record_meta_records, pas_enumerator_record_payload_records, pas_enumerator_record_object_records);
    pas_enumerator_enumerate_all(enumerator);

    pas_enumerator_destroy(enumerator);
}

void testPGMEnumerationAddAndFree() {

    pas_heap_lock_lock();
    pas_root* root = pas_root_create();
    pas_heap_lock_unlock();

    pas_heap_ref heapRef = ISO_HEAP_REF_INITIALIZER_WITH_ALIGNMENT(getpagesize() * 100, getpagesize());
    pas_heap* heap = iso_heap_ref_get_heap(&heapRef);
    pas_physical_memory_transaction transaction;
    pas_physical_memory_transaction_construct(&transaction);

    pas_heap_lock_lock();

    size_t alloc_size = 16384;
    pas_allocation_result result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    result = pas_probabilistic_guard_malloc_allocate(&heap->large_heap, alloc_size, 1, pas_non_compact_allocation_mode, &iso_heap_config, &transaction);
    CHECK(result.begin);

    pas_probabilistic_guard_malloc_deallocate((void*) result.begin);

    pas_heap_lock_unlock();

    pas_enumerator* enumerator = pas_enumerator_create(root, enumeratorReader, nullptr, enumeratorRecorder, nullptr, pas_enumerator_record_meta_records, pas_enumerator_record_payload_records, pas_enumerator_record_object_records);
    pas_enumerator_enumerate_all(enumerator);

    pas_enumerator_destroy(enumerator);

}


} // end namespace

void addEnumerationTests() {
    ADD_TEST(testBasicEnumeration());
    ADD_TEST(testPGMEnumerationBasic());
    ADD_TEST(testPGMEnumerationAddAndFree());
}
