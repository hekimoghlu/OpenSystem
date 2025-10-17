/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include <bmalloc/bmalloc.h>
#include <bmalloc/Environment.h>
#include <bmalloc/IsoHeapInlines.h>
#include <cmath>
#include <cstdlib>
#include <set>
#include <vector>

using namespace bmalloc;
using namespace bmalloc::api;

// We don't have a NO_RETURN_DUE_TO_EXIT, nor should we. That's ridiculous.
static bool hiddenTruthBecauseNoReturnIsStupid() { return true; }

static void usage()
{
    puts("Usage: testb3 [<filter>]");
    if (hiddenTruthBecauseNoReturnIsStupid())
        exit(1);
}

#define RUN(test) do {                          \
        if (!shouldRun(#test))                  \
            break;                              \
        puts(#test "...");                      \
        test;                                   \
        puts(#test ": OK!");                    \
    } while (false)

// Nothing fancy for now; we just use the existing WTF assertion machinery.
#define CHECK(x) do {                                                   \
        if (!!(x))                                                      \
            break;                                                      \
        fprintf(stderr, "%s:%d: in %s: assertion %s failed.\n",         \
            __FILE__, __LINE__, __PRETTY_FUNCTION__, #x);               \
        abort();                                                        \
    } while (false)

static std::set<void*> toptrset(const std::vector<void*>& ptrs)
{
    std::set<void*> result;
    for (void* ptr : ptrs) {
        if (ptr)
            result.insert(ptr);
    }
    return result;
}

static void assertEmptyPointerSet(const std::set<void*>& pointers)
{
    if (PerProcess<Environment>::get()->isDebugHeapEnabled()) {
        printf("    skipping checks because DebugHeap.\n");
        return;
    }
    if (pointers.empty())
        return;
    printf("Pointer set not empty!\n");
    printf("Pointers:");
    for (void* ptr : pointers)
        printf(" %p", ptr);
    printf("\n");
    CHECK(pointers.empty());
}

template<typename heapType>
static void assertHasObjects(IsoHeap<heapType>& heap, std::set<void*> pointers)
{
    if (PerProcess<Environment>::get()->isDebugHeapEnabled()) {
        printf("    skipping checks because DebugHeap.\n");
        return;
    }
    auto& impl = heap.impl();
    std::lock_guard<Mutex> locker(impl.lock);
    impl.forEachLiveObject(
        [&] (void* object) {
            pointers.erase(object);
        });
    assertEmptyPointerSet(pointers);
}

template<typename heapType>
static void assertHasOnlyObjects(IsoHeap<heapType>& heap, std::set<void*> pointers)
{
    if (PerProcess<Environment>::get()->isDebugHeapEnabled()) {
        printf("    skipping checks because DebugHeap.\n");
        return;
    }
    auto& impl = heap.impl();
    std::lock_guard<Mutex> locker(impl.lock);
    impl.forEachLiveObject(
        [&] (void* object) {
            CHECK(pointers.erase(object) == 1);
        });
    assertEmptyPointerSet(pointers);
}

template<typename heapType>
static void assertClean(IsoHeap<heapType>& heap)
{
    scavengeThisThread();
    if (!PerProcess<Environment>::get()->isDebugHeapEnabled()) {
        auto& impl = heap.impl();
        {
            std::lock_guard<Mutex> locker(impl.lock);
            CHECK(!impl.numLiveObjects());
        }
    }
    heap.scavenge();
    if (!PerProcess<Environment>::get()->isDebugHeapEnabled()) {
        auto& impl = heap.impl();
        std::lock_guard<Mutex> locker(impl.lock);
        CHECK(!impl.numCommittedPages());
    }
}

static void testIsoSimple()
{
    static IsoHeap<double> heap;
    void* ptr1 = heap.allocate();
    CHECK(ptr1);
    void* ptr2 = heap.allocate();
    CHECK(ptr2);
    CHECK(ptr1 != ptr2);
    CHECK(std::abs(static_cast<char*>(ptr1) - static_cast<char*>(ptr2)) >= 8);
    assertHasObjects(heap, {ptr1, ptr2});
    heap.deallocate(ptr1);
    heap.deallocate(ptr2);
    assertClean(heap);
}

static void testIsoSimpleScavengeBeforeDealloc()
{
    static IsoHeap<double> heap;
    void* ptr1 = heap.allocate();
    CHECK(ptr1);
    void* ptr2 = heap.allocate();
    CHECK(ptr2);
    CHECK(ptr1 != ptr2);
    CHECK(std::abs(static_cast<char*>(ptr1) - static_cast<char*>(ptr2)) >= 8);
    scavengeThisThread();
    assertHasOnlyObjects(heap, {ptr1, ptr2});
    heap.deallocate(ptr1);
    heap.deallocate(ptr2);
    assertClean(heap);
}

static void testIsoFlipFlopFragmentedPages()
{
    static IsoHeap<double> heap;
    std::vector<void*> ptrs;
    for (unsigned i = 100000; i--;) {
        void* ptr = heap.allocate();
        CHECK(ptr);
        ptrs.push_back(ptr);
    }
    for (unsigned i = 0; i < ptrs.size(); i += 2) {
        heap.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }
    for (unsigned i = ptrs.size() / 2; i--;)
        ptrs.push_back(heap.allocate());
    for (void* ptr : ptrs)
        heap.deallocate(ptr);
    assertClean(heap);
}

static void testIsoFlipFlopFragmentedPagesScavengeInMiddle()
{
    static IsoHeap<double> heap;
    std::vector<void*> ptrs;
    for (unsigned i = 100000; i--;) {
        void* ptr = heap.allocate();
        CHECK(ptr);
        ptrs.push_back(ptr);
    }
    CHECK(toptrset(ptrs).size() == ptrs.size());
    for (unsigned i = 0; i < ptrs.size(); i += 2) {
        heap.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }
    heap.scavenge();
    unsigned numCommittedPagesBefore;
    auto& impl = heap.impl();
    {
        std::lock_guard<Mutex> locker(impl.lock);
        numCommittedPagesBefore = impl.numCommittedPages();
    }
    assertHasOnlyObjects(heap, toptrset(ptrs));
    for (unsigned i = ptrs.size() / 2; i--;)
        ptrs.push_back(heap.allocate());
    {
        std::lock_guard<Mutex> locker(impl.lock);
        CHECK(numCommittedPagesBefore == impl.numCommittedPages());
    }
    for (void* ptr : ptrs)
        heap.deallocate(ptr);
    assertClean(heap);
}

static void testIsoFlipFlopFragmentedPagesScavengeInMiddle288()
{
    static IsoHeap<char[288]> heap;
    std::vector<void*> ptrs;
    for (unsigned i = 100000; i--;) {
        void* ptr = heap.allocate();
        CHECK(ptr);
        ptrs.push_back(ptr);
    }
    CHECK(toptrset(ptrs).size() == ptrs.size());
    for (unsigned i = 0; i < ptrs.size(); i += 2) {
        heap.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }
    heap.scavenge();
    unsigned numCommittedPagesBefore;
    auto& impl = heap.impl();
    {
        std::lock_guard<Mutex> locker(impl.lock);
        numCommittedPagesBefore = impl.numCommittedPages();
    }
    assertHasOnlyObjects(heap, toptrset(ptrs));
    for (unsigned i = ptrs.size() / 2; i--;)
        ptrs.push_back(heap.allocate());
    {
        std::lock_guard<Mutex> locker(impl.lock);
        CHECK(numCommittedPagesBefore == impl.numCommittedPages());
    }
    for (void* ptr : ptrs)
        heap.deallocate(ptr);
    assertClean(heap);
}

class BisoMalloced {
    MAKE_BISO_MALLOCED(BisoMalloced);
public:
    BisoMalloced(int x, float y)
        : x(x)
        , y(y)
    {
    }
    
    int x;
    float y;
};

MAKE_BISO_MALLOCED_IMPL(BisoMalloced);

static void testBisoMalloced()
{
    BisoMalloced* ptr = new BisoMalloced(4, 5);
    assertHasObjects(BisoMalloced::bisoHeap(), { ptr });
    delete ptr;
    assertClean(BisoMalloced::bisoHeap());
}

class BisoMallocedInline {
    MAKE_BISO_MALLOCED_INLINE(BisoMalloced);
public:
    BisoMallocedInline(int x, float y)
        : x(x)
        , y(y)
    {
    }
    
    int x;
    float y;
};

static void testBisoMallocedInline()
{
    BisoMallocedInline* ptr = new BisoMallocedInline(4, 5);
    assertHasObjects(BisoMallocedInline::bisoHeap(), { ptr });
    delete ptr;
    assertClean(BisoMallocedInline::bisoHeap());
}

static void run(const char* filter)
{
    auto shouldRun = [&] (const char* testName) -> bool {
        return !filter || !!strcasestr(testName, filter);
    };
    
    RUN(testIsoSimple());
    RUN(testIsoSimpleScavengeBeforeDealloc());
    RUN(testIsoFlipFlopFragmentedPages());
    RUN(testIsoFlipFlopFragmentedPagesScavengeInMiddle());
    RUN(testIsoFlipFlopFragmentedPagesScavengeInMiddle288());
    RUN(testBisoMalloced());
    RUN(testBisoMallocedInline());
    
    puts("Success!");
}

int main(int argc, char** argv)
{
    const char* filter = nullptr;
    switch (argc) {
    case 1:
        break;
    case 2:
        filter = argv[1];
        break;
    default:
        usage();
        break;
    }
    
    run(filter);
    return 0;
}

