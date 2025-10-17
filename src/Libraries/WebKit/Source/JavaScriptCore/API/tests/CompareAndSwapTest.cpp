/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "config.h"
#include "CompareAndSwapTest.h"

#include <functional>
#include <stdio.h>
#include <wtf/Atomics.h>
#include <wtf/Threading.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

class Bitmap {
public:
    Bitmap() { clearAll(); }

    inline void clearAll();
    inline bool concurrentTestAndSet(size_t n);
    inline size_t numBits() const { return words * wordSize; }

private:
    static constexpr size_t Size = 4096*10;

    static constexpr unsigned wordSize = sizeof(uint8_t) * 8;
    static constexpr unsigned words = (Size + wordSize - 1) / wordSize;
    static constexpr uint8_t one = 1;

    uint8_t bits[words];
};

inline void Bitmap::clearAll()
{
    memset(&bits, 0, sizeof(bits));
}

inline bool Bitmap::concurrentTestAndSet(size_t n)
{
    uint8_t mask = one << (n % wordSize);
    size_t index = n / wordSize;
    uint8_t* wordPtr = &bits[index];
    uint8_t oldValue;
    do {
        oldValue = *wordPtr;
        if (oldValue & mask)
            return true;
    } while (!WTF::atomicCompareExchangeWeakRelaxed(wordPtr, oldValue, static_cast<uint8_t>(oldValue | mask)));
    return false;
}

struct Data {
    Bitmap* bitmap;
    int id;
    int numThreads;
};

static void setBitThreadFunc(void* p)
{
    Data* data = reinterpret_cast<Data*>(p);
    Bitmap* bitmap = data->bitmap;
    size_t numBits = bitmap->numBits();

    // The computed start index here is heuristic that seems to maximize (anecdotally)
    // the chance for the CAS issue to manifest.
    size_t start = (numBits * (data->numThreads - data->id)) / data->numThreads;

    printf("   started Thread %d\n", data->id);
    for (size_t i = start; i < numBits; i++)
        while (!bitmap->concurrentTestAndSet(i)) { }
    for (size_t i = 0; i < start; i++)
        while (!bitmap->concurrentTestAndSet(i)) { }

    printf("   finished Thread %d\n", data->id);
}

void testCompareAndSwap()
{
    Bitmap bitmap;
    const int numThreads = 5;
    RefPtr<Thread> threads[numThreads];
    Data data[numThreads];

    WTF::initialize();
    
    printf("Starting %d threads for CompareAndSwap test.  Test should complete without hanging.\n", numThreads);
    for (int i = 0; i < numThreads; i++) {
        data[i].bitmap = &bitmap;
        data[i].id = i;
        data[i].numThreads = numThreads;
        threads[i] = Thread::create("setBitThreadFunc"_s, std::bind(setBitThreadFunc, &data[i]));
    }

    printf("Waiting for %d threads to join\n", numThreads);
    for (int i = 0; i < numThreads; i++)
        threads[i]->waitForCompletion();

    printf("PASS: CompareAndSwap test completed without a hang\n");
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
