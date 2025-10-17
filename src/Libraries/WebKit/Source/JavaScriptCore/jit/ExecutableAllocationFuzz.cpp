/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include "ExecutableAllocationFuzz.h"

#include "TestRunnerUtils.h"
#include <wtf/Atomics.h>
#include <wtf/DataLog.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakRandom.h>

namespace JSC {

static Atomic<unsigned> s_numberOfExecutableAllocationFuzzChecks;
unsigned numberOfExecutableAllocationFuzzChecks()
{
    return s_numberOfExecutableAllocationFuzzChecks.load();
}

ExecutableAllocationFuzzResult doExecutableAllocationFuzzing()
{
    ASSERT(Options::useExecutableAllocationFuzz());

    if (Options::fireExecutableAllocationFuzzRandomly()) {
        static LazyNeverDestroyed<WeakRandom> random;
        static std::once_flag once;
        std::call_once(once, [] () {
            random.construct();
        });

        static Lock fuzzingLock;
        Locker locker { fuzzingLock };
        
        if (random->returnTrueWithProbability(Options::fireExecutableAllocationFuzzRandomlyProbability()))
            return PretendToFailExecutableAllocation;

        return AllowNormalExecutableAllocation;
    }
    
    unsigned oldValue;
    unsigned newValue;
    do {
        oldValue = s_numberOfExecutableAllocationFuzzChecks.load();
        newValue = oldValue + 1;
    } while (!s_numberOfExecutableAllocationFuzzChecks.compareExchangeWeak(oldValue, newValue));
    
    if (newValue == Options::fireExecutableAllocationFuzzAt()) {
        if (Options::verboseExecutableAllocationFuzz()) {
            dataLog("Will pretend to fail executable allocation.\n");
            WTFReportBacktrace();
        }
        return PretendToFailExecutableAllocation;
    }
    
    if (Options::fireExecutableAllocationFuzzAtOrAfter()
        && newValue >= Options::fireExecutableAllocationFuzzAtOrAfter()) {
        if (Options::verboseExecutableAllocationFuzz()) {
            dataLog("Will pretend to fail executable allocation.\n");
            WTFReportBacktrace();
        }
        return PretendToFailExecutableAllocation;
    }
    
    return AllowNormalExecutableAllocation;
}

} // namespace JSC

