/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#pragma once

#include "SamplingCounter.h"
#include <wtf/Assertions.h>

namespace JSC {

// This allows the JIT to distinguish between uses of the barrier for different
// kinds of writes. This is used by the JIT for profiling, and may be appropriate
// for allowing the GC implementation to specialize the JIT's write barrier code
// for different kinds of target objects.
enum WriteBarrierUseKind {
    // This allows specialization for access to the property storage (either
    // array element or property), but not for any other kind of property
    // accesses (such as writes that are a consequence of setter execution).
    WriteBarrierForPropertyAccess,
    
    // This allows specialization for variable accesses (such as global or
    // scoped variables).
    WriteBarrierForVariableAccess,
    
    // This captures all other forms of write barriers. It should always be
    // correct to use a generic access write barrier, even when storing to
    // properties. Hence, if optimization is not necessary, it is preferable
    // to just use a generic access.
    WriteBarrierForGenericAccess
};

class WriteBarrierCounters {
private:
    WriteBarrierCounters() { }

public:
#if ENABLE(WRITE_BARRIER_PROFILING)
    static GlobalSamplingCounter usesWithBarrierFromCpp;
    static GlobalSamplingCounter usesWithoutBarrierFromCpp;
    static GlobalSamplingCounter usesWithBarrierFromJit;
    static GlobalSamplingCounter usesForPropertiesFromJit;
    static GlobalSamplingCounter usesForVariablesFromJit;
    static GlobalSamplingCounter usesWithoutBarrierFromJit;
    
    static void initialize();
    
    static GlobalSamplingCounter& jitCounterFor(WriteBarrierUseKind useKind)
    {
        switch (useKind) {
        case WriteBarrierForPropertyAccess:
            return usesForPropertiesFromJit;
        case WriteBarrierForVariableAccess:
            return usesForVariablesFromJit;
        default:
            ASSERT(useKind == WriteBarrierForGenericAccess);
            return usesWithBarrierFromJit;
        }
    }
#else
    // These are necessary to work around not having conditional exports.
    JS_EXPORT_PRIVATE static char usesWithBarrierFromCpp;
    JS_EXPORT_PRIVATE static char usesWithoutBarrierFromCpp;
#endif // ENABLE(WRITE_BARRIER_PROFILING)

    static void countWriteBarrier()
    {
#if ENABLE(WRITE_BARRIER_PROFILING)
        WriteBarrierCounters::usesWithBarrierFromCpp.count();
#endif
    }
};

} // namespace JSC
