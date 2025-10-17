/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include "SuperSampler.h"

#include "Options.h"
#include <wtf/DataLog.h>
#include <wtf/Lock.h>
#include <wtf/Seconds.h>
#include <wtf/Threading.h>

namespace JSC {

std::atomic<uint32_t> g_superSamplerCount;
std::atomic<bool> g_superSamplerEnabled;

static Lock lock;
static double in WTF_GUARDED_BY_LOCK(lock);
static double out WTF_GUARDED_BY_LOCK(lock);

void initializeSuperSampler()
{
    if (!Options::useSuperSampler())
        return;

    Thread::create(
        "JSC Super Sampler"_s,
        [] () {
            const int sleepQuantum = 3;
            const int printingPeriod = 3000;
            for (;;) {
                for (int ms = 0; ms < printingPeriod; ms += sleepQuantum) {
                    if (g_superSamplerEnabled) {
                        Locker locker { lock };
                        if (g_superSamplerCount)
                            in++;
                        else
                            out++;
                    }
                    if (sleepQuantum)
                        sleep(Seconds::fromMilliseconds(sleepQuantum));
                }
                printSuperSamplerState();
                if (static_cast<int32_t>(g_superSamplerCount) < 0)
                    dataLog("WARNING: Super sampler undercount detected!\n");
            }
        });
}

void resetSuperSamplerState()
{
    Locker locker { lock };
    in = 0;
    out = 0;
}

void printSuperSamplerState()
{
    if (!Options::useSuperSampler())
        return;

    Locker locker { lock };
    double percentage = 100.0 * in / (in + out);
    if (percentage != percentage)
        percentage = 0.0;
    dataLog("Percent time behind super sampler flag: ", percentage, "%\n");
}

void enableSuperSampler()
{
    Locker locker { lock };
    g_superSamplerEnabled = true;
}

void disableSuperSampler()
{
    Locker locker { lock };
    g_superSamplerEnabled = false;
}

} // namespace JSC

