/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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

#include "DFGCommon.h"

#if ENABLE(FTL_JIT)

#include "CCallHelpers.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC { namespace FTL {

class State;
struct OSRExit;

// This is an object that stores some interesting data about an OSR exit. It's expected that you will
// scrape this data from this object by the time compilation finishes.
struct OSRExitHandle : public ThreadSafeRefCounted<OSRExitHandle> {
    OSRExitHandle(unsigned index, JITCode* jitCode)
        : m_index(index)
        , m_jitCode(jitCode)
    {
    }

    unsigned m_index;
    JITCode* m_jitCode;

    // This is the label at which the OSR exit jump lives. This will get populated once the OSR exit
    // emits its jump. This happens immediately when you call OSRExit::appendOSRExit(). It happens at
    // some time during late path emission if you do OSRExit::appendOSRExitLater().
    CCallHelpers::Label label;

    // This emits the exit thunk and populates 'label'.
    void emitExitThunk(State&, CCallHelpers&);
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
