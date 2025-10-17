/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
// On Mac, you can build this like so:
// xcrun clang++ -o LockFairnessTest Source/WTF/benchmarks/LockFairnessTest.cpp -O3 -W -ISource/WTF -ISource/WTF/icu -ISource/WTF/benchmarks -LWebKitBuild/Release -lWTF -framework Foundation -licucore -std=c++14 -fvisibility=hidden

#include "config.h"

#include "ToyLocks.h"
#include <thread>
#include <unistd.h>
#include <wtf/CommaPrinter.h>
#include <wtf/Compiler.h>
#include <wtf/DataLog.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ParkingLot.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Threading.h>
#include <wtf/ThreadingPrimitives.h>
#include <wtf/Vector.h>
#include <wtf/WordLock.h>
#include <wtf/text/CString.h>

namespace {

NO_RETURN void usage()
{
    printf("Usage: LockFairnessTest yieldspinlock|pausespinlock|wordlock|lock|barginglock|bargingwordlock|thunderlock|thunderwordlock|cascadelock|cascadewordlockhandofflock|unfairlock|mutex|all <num threads> <seconds per test> <microseconds in critical section>\n");
    exit(1);
}

unsigned numThreads;
double secondsPerTest;
unsigned microsecondsInCriticalSection;

struct Benchmark {
    template<typename LockType>
    static void run(const char* name)
    {
        LockType lock;
        std::unique_ptr<unsigned[]> counts = makeUniqueWithoutFastMallocCheck<unsigned[]>(numThreads);
        std::unique_ptr<RefPtr<Thread>[]> threads = makeUniqueWithoutFastMallocCheck<RefPtr<Thread>[]>(numThreads);
    
        volatile bool keepGoing = true;
    
        lock.lock();
    
        for (unsigned threadIndex = numThreads; threadIndex--;) {
            counts[threadIndex] = 0;
            threads[threadIndex] = Thread::create(
                "Benchmark Thread"_s,
                [&, threadIndex] () {
                    if (!microsecondsInCriticalSection) {
                        while (keepGoing) {
                            lock.lock();
                            counts[threadIndex]++;
                            lock.unlock();
                        }
                        return;
                    }
                    
                    while (keepGoing) {
                        lock.lock();
                        counts[threadIndex]++;
                        usleep(microsecondsInCriticalSection);
                        lock.unlock();
                    }
                });
        }
    
        sleep(100_ms);
        lock.unlock();
    
        sleep(Seconds { secondsPerTest });
    
        keepGoing = false;
        lock.lock();
    
        dataLog(name, ": ");
        CommaPrinter comma;
        for (unsigned threadIndex = numThreads; threadIndex--;)
            dataLog(comma, counts[threadIndex]);
        dataLog("\n");
    
        lock.unlock();
        for (unsigned threadIndex = numThreads; threadIndex--;)
            threads[threadIndex]->waitForCompletion();
    }
};

} // anonymous namespace

int main(int argc, char** argv)
{
    WTF::initialize();
    
    if (argc != 5
        || sscanf(argv[2], "%u", &numThreads) != 1
        || sscanf(argv[3], "%lf", &secondsPerTest) != 1
        || sscanf(argv[4], "%u", &microsecondsInCriticalSection) != 1)
        usage();
    
    runEverything<Benchmark>(argv[1]);
    
    return 0;
}
