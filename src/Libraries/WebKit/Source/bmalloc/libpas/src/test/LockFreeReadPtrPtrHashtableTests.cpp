/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#include "pas_heap_lock.h"
#include "pas_lock.h"
#include "pas_lock_free_read_ptr_ptr_hashtable.h"
#include <thread>
#include <vector>
#include <map>

using namespace std;

namespace {

unsigned hashFunction(const void* key, void* arg)
{
    PAS_UNUSED_PARAM(arg);
    return pas_hash_ptr(key);
}

void testChaos(unsigned numHitReaderThreads,
               unsigned numMissReaderThreads,
               unsigned numWriterThreads,
               unsigned numAdditions)
{
    pas_lock_free_read_ptr_ptr_hashtable table;
    map<unsigned, unsigned> ourMap;
    pas_lock mapLock;
    vector<unsigned> ourList;
    pas_lock listLock;
    bool done;
    vector<thread> writerThreads;
    vector<thread> readerThreads;
    pas_lock stopLock;
    unsigned numThreadsStopped;

    table = PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_INITIALIZER;
    done = false;

    mapLock = PAS_LOCK_INITIALIZER;
    listLock = PAS_LOCK_INITIALIZER;

    stopLock = PAS_LOCK_INITIALIZER;
    numThreadsStopped = 0;

    auto threadIsStopping =
        [&] () {
            pas_lock_lock(&stopLock);
            numThreadsStopped++;
            pas_lock_unlock(&stopLock);
        };
    
    auto writerThreadFunc =
        [&] () {
            for (unsigned count = numAdditions; count--;) {
                unsigned key;
                unsigned value;

                value = deterministicRandomNumber(UINT_MAX);

                pas_lock_lock(&mapLock);
                for (;;) {
                    key = deterministicRandomNumber(UINT_MAX);
                    if (!ourMap.count(key))
                        break;
                }
                ourMap[key] = value;
                pas_lock_unlock(&mapLock);

                pas_heap_lock_lock();
                pas_lock_free_read_ptr_ptr_hashtable_set(
                    &table,
                    hashFunction,
                    NULL,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(key)),
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(value)),
                    pas_lock_free_read_ptr_ptr_hashtable_add_new);
                pas_heap_lock_unlock();

                pas_lock_lock(&listLock);
                ourList.push_back(key);
                pas_lock_unlock(&listLock);
            }
            threadIsStopping();
        };

    auto hitReaderThreadFunc =
        [&] () {
            while (!done) {
                unsigned key;

                pas_lock_lock(&listLock);
                if (ourList.empty()) {
                    pas_lock_unlock(&listLock);
                    continue;
                }
                key = ourList[deterministicRandomNumber(static_cast<unsigned>(ourList.size()))];
                pas_lock_unlock(&listLock);

                const void* result = pas_lock_free_read_ptr_ptr_hashtable_find(
                    &table, hashFunction, NULL,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(key)));

                pas_lock_lock(&mapLock);
                unsigned value = ourMap[key];
                CHECK_EQUAL(result, reinterpret_cast<const void*>(static_cast<uintptr_t>(value)));
                pas_lock_unlock(&mapLock);
            }
            threadIsStopping();
        };

    auto missReaderThreadFunc =
        [&] () {
            while (!done) {
                uint64_t key;

                key = static_cast<uint64_t>(deterministicRandomNumber(UINT_MAX)) +
                    static_cast<uint64_t>(UINT_MAX);

                const void* result = pas_lock_free_read_ptr_ptr_hashtable_find(
                    &table, hashFunction, NULL,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(key)));

                CHECK(!result);
            }
            threadIsStopping();
        };

    for (unsigned count = numHitReaderThreads; count--;)
        readerThreads.push_back(thread(hitReaderThreadFunc));
    for (unsigned count = numMissReaderThreads; count--;)
        readerThreads.push_back(thread(missReaderThreadFunc));

    for (unsigned count = numWriterThreads; count--;)
        writerThreads.push_back(thread(writerThreadFunc));

    for (thread& thread : writerThreads)
        thread.join();

    done = true;

    for (thread& thread : readerThreads)
        thread.join();

    CHECK_EQUAL(numThreadsStopped, readerThreads.size() + writerThreads.size());
}

} // anonymous namespace

void addLockFreeReadPtrPtrHashtableTests()
{
    ADD_TEST(testChaos(1, 1, 1, 2000000));
    ADD_TEST(testChaos(10, 10, 10, 200000));
}

