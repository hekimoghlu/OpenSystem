/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#include "iso_heap.h"
#include "pas_fast_tls.h"
#include "pas_scavenger.h"
#include <pthread.h>
#include <vector>
#include <thread>

using namespace std;

namespace {

vector<pthread_key_t> keys;

pas_heap_ref isoHeap = ISO_HEAP_REF_INITIALIZER(32);
pas_primitive_heap_ref isoPrimitiveHeap = ISO_PRIMITIVE_HEAP_REF_INITIALIZER;

void destructor(void* value)
{
    PAS_UNUSED_PARAM(value);
    for (pthread_key_t key : keys)
        pthread_setspecific(key, "infinite loop");
    iso_deallocate(iso_allocate_common_primitive(666, pas_non_compact_allocation_mode));
    iso_deallocate(iso_reallocate_common_primitive(
                       iso_allocate_common_primitive(666, pas_non_compact_allocation_mode), 1337, pas_reallocate_free_if_successful, pas_non_compact_allocation_mode));
    iso_deallocate(iso_allocate(&isoHeap, pas_non_compact_allocation_mode));
    iso_deallocate(iso_allocate_array_by_count(&isoHeap, 100, 1, pas_non_compact_allocation_mode));
    iso_deallocate(iso_allocate_array_by_count(&isoHeap, 100, 64, pas_non_compact_allocation_mode));
    iso_deallocate(iso_reallocate_array_by_count(
                       iso_allocate(&isoHeap, pas_non_compact_allocation_mode), &isoHeap, 200, pas_reallocate_free_if_successful, pas_non_compact_allocation_mode));
    iso_deallocate(iso_allocate_primitive(&isoPrimitiveHeap, 666, pas_non_compact_allocation_mode));
    iso_deallocate(iso_allocate_primitive_with_alignment(&isoPrimitiveHeap, 128, 64, pas_non_compact_allocation_mode));
    iso_deallocate(iso_reallocate_primitive(
                       iso_allocate_primitive(&isoPrimitiveHeap, 666, pas_non_compact_allocation_mode), &isoPrimitiveHeap, 1337,
                       pas_reallocate_free_if_successful, pas_non_compact_allocation_mode));
}

void testTSD(unsigned numKeysBeforeAllocation,
             unsigned numKeysAfterAllocation,
             bool initializeJSCKeysBeforeAllocation,
             bool initializeJSCKeysAfterAllocation,
             unsigned numAllocations,
             unsigned allocationSize)
{
#if PAS_HAVE_PTHREAD_MACHDEP_H
    auto initializeFastKey =
        [&] (int key) {
            pthread_key_init_np(key, destructor);
            _pthread_setspecific_direct(key, const_cast<void*>(static_cast<const void*>("direct")));
            keys.push_back(key);
        };
    
    auto initializeJSCKeys =
        [&] () {
            initializeFastKey(__PTK_FRAMEWORK_COREDATA_KEY5);
            initializeFastKey(__PTK_FRAMEWORK_JAVASCRIPTCORE_KEY0);
            initializeFastKey(__PTK_FRAMEWORK_JAVASCRIPTCORE_KEY1);
            initializeFastKey(__PTK_FRAMEWORK_JAVASCRIPTCORE_KEY2);
            initializeFastKey(__PTK_FRAMEWORK_JAVASCRIPTCORE_KEY3);
        };
#else
    auto initializeJSCKeys = [] () { };
#endif
    
    auto initializeKeys =
        [&] (unsigned numKeys) {
            for (unsigned i = numKeys; i--;) {
                pthread_key_t key;
                pthread_key_create(&key, destructor);
                pthread_setspecific(key, "hello world");
                keys.push_back(key);
            }
        };
    
    thread myThread = thread(
        [&] () {
            if (initializeJSCKeysBeforeAllocation)
                initializeJSCKeys();
            initializeKeys(numKeysBeforeAllocation);
            for (unsigned i = numAllocations; i--;)
                iso_deallocate(iso_allocate_common_primitive(allocationSize, pas_non_compact_allocation_mode));
            if (initializeJSCKeysAfterAllocation)
                initializeJSCKeys();
            initializeKeys(numKeysAfterAllocation);
        });

    myThread.join();

    pas_scavenger_run_synchronously_now();
}

} // anonymous namespace

void addTSDTests()
{
    ForceTLAs forceTLAs;
    DisableBitfit disableBitfit;

    // We try different approaches to this test in case it matters.
    ADD_TEST(testTSD(1, 0, false, false, 100, 100));
    ADD_TEST(testTSD(0, 1, false, false, 100, 100));
    ADD_TEST(testTSD(1, 1, false, false, 100, 100));
    ADD_TEST(testTSD(1, 1, true, false, 100, 100));
    ADD_TEST(testTSD(1, 1, false, true, 100, 100));
    ADD_TEST(testTSD(10, 0, false, false, 100, 100));
    ADD_TEST(testTSD(0, 10, false, false, 100, 100));
    ADD_TEST(testTSD(10, 10, false, false, 100, 100));
    ADD_TEST(testTSD(10, 10, true, false, 100, 100));
    ADD_TEST(testTSD(10, 10, false, true, 100, 100));
    ADD_TEST(testTSD(100, 0, false, false, 100, 100));
    ADD_TEST(testTSD(0, 100, false, false, 100, 100));
    ADD_TEST(testTSD(100, 100, false, false, 100, 100));
    ADD_TEST(testTSD(100, 100, false, false, 1000, 16));
    ADD_TEST(testTSD(100, 100, true, false, 100, 16));
    ADD_TEST(testTSD(100, 100, false, true, 100, 16));
}

