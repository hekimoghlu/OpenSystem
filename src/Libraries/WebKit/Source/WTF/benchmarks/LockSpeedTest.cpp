/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
// xcrun clang++ -o LockSpeedTest Source/WTF/benchmarks/LockSpeedTest.cpp -O3 -W -ISource/WTF -ISource/WTF/icu -ISource/WTF/benchmarks -LWebKitBuild/Release -lWTF -framework Foundation -licucore -std=c++14 -fvisibility=hidden

#include "config.h"

#include "ToyLocks.h"
#include <thread>
#include <unistd.h>
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

unsigned numThreadGroups;
unsigned numThreadsPerGroup;
unsigned workPerCriticalSection;
unsigned workBetweenCriticalSections;
double secondsPerTest;
    
NO_RETURN void usage()
{
    printf("Usage: LockSpeedTest yieldspinlock|pausespinlock|wordlock|lock|barginglock|bargingwordlock|thunderlock|thunderwordlock|cascadelock|cascadewordlock|handofflock|unfairlock|mutex|all <num thread groups> <num threads per group> <work per critical section> <work between critical sections> <spin limit> <seconds per test>\n");
    exit(1);
}

template<typename Type>
struct WithPadding {
    Type value;
    char buf[300]; // It's best if this isn't perfect to avoid false sharing.
};

HashMap<CString, Vector<double>> results;

void reportResult(const char* name, double value)
{
    dataLogF("%s: %.3lf KHz\n", name, value);
    results.add(name, Vector<double>()).iterator->value.append(value);
}

struct Benchmark {
    template<typename LockType>
    static void run(const char* name)
    {
        std::unique_ptr<WithPadding<LockType>[]> locks = makeUniqueWithoutFastMallocCheck<WithPadding<LockType>[]>(numThreadGroups);
        std::unique_ptr<WithPadding<double>[]> words = makeUniqueWithoutFastMallocCheck<WithPadding<double>[]>(numThreadGroups);
        std::unique_ptr<RefPtr<Thread>[]> threads = makeUniqueWithoutFastMallocCheck<RefPtr<Thread>[]>(numThreadGroups * numThreadsPerGroup);

        volatile bool keepGoing = true;

        MonotonicTime before = MonotonicTime::now();
    
        Lock numIterationsLock;
        uint64_t numIterations = 0;
    
        for (unsigned threadGroupIndex = numThreadGroups; threadGroupIndex--;) {
            words[threadGroupIndex].value = 0;

            for (unsigned threadIndex = numThreadsPerGroup; threadIndex--;) {
                threads[threadGroupIndex * numThreadsPerGroup + threadIndex] = Thread::create(
                    "Benchmark thread"_s,
                    [threadGroupIndex, &locks, &words, &keepGoing, &numIterationsLock, &numIterations] () {
                        double localWord = 0;
                        double value = 1;
                        unsigned myNumIterations = 0;
                        while (keepGoing) {
                            locks[threadGroupIndex].value.lock();
                            for (unsigned j = workPerCriticalSection; j--;) {
                                words[threadGroupIndex].value += value;
                                value = words[threadGroupIndex].value;
                            }
                            locks[threadGroupIndex].value.unlock();
                            for (unsigned j = workBetweenCriticalSections; j--;) {
                                localWord += value;
                                value = localWord;
                            }
                            myNumIterations++;
                        }
                        Locker locker { numIterationsLock };
                        numIterations += myNumIterations;
                    });
            }
        }

        sleep(Seconds { secondsPerTest });
        keepGoing = false;
    
        for (unsigned threadIndex = numThreadGroups * numThreadsPerGroup; threadIndex--;)
            threads[threadIndex]->waitForCompletion();

        MonotonicTime after = MonotonicTime::now();
    
        reportResult(name, numIterations / (after - before).seconds() / 1000);
    }
};

unsigned rangeMin;
unsigned rangeMax;
unsigned rangeStep;
unsigned* rangeVariable;

bool parseValue(const char* string, unsigned* variable)
{
    unsigned myRangeMin;
    unsigned myRangeMax;
    unsigned myRangeStep;
    if (sscanf(string, "%u-%u:%u", &myRangeMin, &myRangeMax, &myRangeStep) == 3) {
        if (rangeVariable) {
            fprintf(stderr, "Can only have one variable with a range.\n");
            return false;
        }
        
        rangeMin = myRangeMin;
        rangeMax = myRangeMax;
        rangeStep = myRangeStep;
        rangeVariable = variable;
        return true;
    }
    
    if (sscanf(string, "%u", variable) == 1)
        return true;
    
    return false;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    WTF::initialize();
    
    if (argc != 8
        || !parseValue(argv[2], &numThreadGroups)
        || !parseValue(argv[3], &numThreadsPerGroup)
        || !parseValue(argv[4], &workPerCriticalSection)
        || !parseValue(argv[5], &workBetweenCriticalSections)
        || !parseValue(argv[6], &toyLockSpinLimit)
        || sscanf(argv[7], "%lf", &secondsPerTest) != 1)
        usage();
    
    if (rangeVariable) {
        dataLog("Running with rangeMin = ", rangeMin, ", rangeMax = ", rangeMax, ", rangeStep = ", rangeStep, "\n");
        for (unsigned value = rangeMin; value <= rangeMax; value += rangeStep) {
            dataLog("Running with value = ", value, "\n");
            *rangeVariable = value;
            runEverything<Benchmark>(argv[1]);
        }
    } else
        runEverything<Benchmark>(argv[1]);
    
    for (auto& entry : results) {
        printf("%s = {", entry.key.data());
        bool first = true;
        for (double value : entry.value) {
            if (first)
                first = false;
            else
                printf(", ");
            printf("%.3lf", value);
        }
        printf("};\n");
    }

    return 0;
}
