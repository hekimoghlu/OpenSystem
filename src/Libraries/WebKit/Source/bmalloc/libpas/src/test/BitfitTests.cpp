/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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

#if PAS_ENABLE_ISO

#include "iso_heap.h"
#include "iso_heap_innards.h"
#include "pas_bitfit_heap.h"
#include "pas_bitfit_size_class.h"
#include "pas_heap.h"
#include <vector>

using namespace std;

namespace {

vector<size_t> getBitfitSizeClasses()
{
    vector<size_t> result;
    
    pas_bitfit_heap* heap = pas_compact_atomic_bitfit_heap_ptr_load_non_null(
        &iso_common_primitive_heap.segregated_heap.bitfit_heap);

    pas_bitfit_page_config_variant variant;
    for (PAS_EACH_BITFIT_PAGE_CONFIG_VARIANT_ASCENDING(variant)) {
        pas_bitfit_directory* directory = pas_bitfit_heap_get_directory(heap, variant);

        for (pas_bitfit_size_class* sizeClass =
                 pas_compact_atomic_bitfit_size_class_ptr_load(&directory->largest_size_class);
             sizeClass;
             sizeClass = pas_compact_atomic_bitfit_size_class_ptr_load(&sizeClass->next_smaller))
            result.push_back(sizeClass->size);
    }

    sort(result.begin(), result.end());

    return result;
}

template<typename... Arguments>
void assertSizeClasses(Arguments... sizeClasses)
{
    vector<size_t> expectedSizeClasses { sizeClasses... };
    vector<size_t> actualSizeClasses = getBitfitSizeClasses();

    if (actualSizeClasses.size() == expectedSizeClasses.size()) {
        bool allGood = true;
        for (size_t index = 0; index < actualSizeClasses.size(); ++index) {
            if (actualSizeClasses[index] != expectedSizeClasses[index]) {
                allGood = false;
                break;
            }
        }
        if (allGood)
            return;
    }
    
    cout << "Expected size classes: ";
    for (size_t index = 0; index < expectedSizeClasses.size(); ++index) {
        if (index)
            cout << ", ";
        cout << expectedSizeClasses[index];
    }
    cout << "\n";
    cout << "But got: ";
    for (size_t index = 0; index < actualSizeClasses.size(); ++index) {
        if (index)
            cout << ", ";
        cout << actualSizeClasses[index];
    }
    cout << "\n";
    CHECK(!"Test failed");
}

void testAllocateAlignedSmallerThanSizeClassAndSmallerThanLargestAvailable(
    size_t firstSize, size_t numFirstObjects, size_t indexOfObjectToFree,
    size_t fillerObjectSize, size_t alignedSize, size_t numAlignedObjects)
{
    static constexpr bool verbose = false;
    
    vector<void*> objects;
    void* freedObject;
    uintptr_t freedObjectBegin;
    uintptr_t freedObjectEnd;

    for (size_t index = 0; index < numFirstObjects; ++index) {
        void* ptr = iso_allocate_common_primitive(firstSize, pas_non_compact_allocation_mode);
        if (verbose)
            cout << "Adding first object " << ptr << "\n";
        objects.push_back(ptr);
    }

    assertSizeClasses(firstSize);

    freedObject = objects[indexOfObjectToFree];
    iso_deallocate(freedObject);

    freedObjectBegin = reinterpret_cast<uintptr_t>(freedObject);
    freedObjectEnd = freedObjectBegin + firstSize;

    vector<void*> fillerObjects;
    bool didStartAllocatingInFreedObject = false;
    for (;;) {
        void* fillerObject = iso_allocate_common_primitive(fillerObjectSize, pas_non_compact_allocation_mode);
        uintptr_t fillerObjectBegin = reinterpret_cast<uintptr_t>(fillerObject);
        uintptr_t fillerObjectEnd = fillerObjectBegin + fillerObjectSize;
        if (verbose)
            cout << "Allocated filler object " << fillerObject << "\n";
        if (fillerObjectBegin >= freedObjectBegin && fillerObjectEnd <= freedObjectEnd) {
            didStartAllocatingInFreedObject = true;
            fillerObjects.push_back(fillerObject);
        } else {
            if (didStartAllocatingInFreedObject)
                break;
        }
    }

    CHECK_EQUAL(fillerObjects.size(), firstSize / fillerObjectSize);
    assertSizeClasses(fillerObjectSize, firstSize);

    for (size_t index = 1; index < fillerObjects.size(); ++index)
        iso_deallocate(fillerObjects[index]);

    for (size_t index = 0; index < numAlignedObjects; ++index) {
        void* ptr = iso_allocate_common_primitive_with_alignment(alignedSize, alignedSize, pas_non_compact_allocation_mode);;
        if (verbose)
            cout << "Allocated aligned object " << ptr << "\n";
    }

    assertSizeClasses(fillerObjectSize, firstSize);
}

} // anonymous namespace

#endif // PAS_ENABLE_ISO

void addBitfitTests()
{
#if PAS_ENABLE_ISO
    ForceBitfit forceBitfit;
    
    ADD_TEST(testAllocateAlignedSmallerThanSizeClassAndSmallerThanLargestAvailable(
                 1056, 100, 0, 16, 1024, 100));
#endif // PAS_ENABLE_ISO
}
