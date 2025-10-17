/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "DFGCommon.h"

#include <wtf/Lock.h>
#include <wtf/PrintStream.h>

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

const char* const tierName = "DFG ";

static Lock crashLock;

// Use WTF_IGNORES_THREAD_SAFETY_ANALYSIS since the function keeps holding the lock when returning.
void startCrashing() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
{
    crashLock.lock();
}

bool isCrashing()
{
    return crashLock.isLocked();
}

bool stringLessThan(StringImpl& a, StringImpl& b)
{
    unsigned minLength = std::min(a.length(), b.length());
    for (unsigned i = 0; i < minLength; ++i) {
        if (a[i] == b[i])
            continue;
        return a[i] < b[i];
    }
    return a.length() < b.length();
}

} } // namespace JSC::DFG

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, OptimizationFixpointState state)
{
    switch (state) {
    case BeforeFixpoint:
        out.print("BeforeFixpoint");
        return;
    case FixpointNotConverged:
        out.print("FixpointNotConverged");
        return;
    case FixpointConverged:
        out.print("FixpointConverged");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, GraphForm form)
{
    switch (form) {
    case LoadStore:
        out.print("LoadStore");
        return;
    case ThreadedCPS:
        out.print("ThreadedCPS");
        return;
    case SSA:
        out.print("SSA");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, UnificationState state)
{
    switch (state) {
    case LocallyUnified:
        out.print("LocallyUnified");
        return;
    case GloballyUnified:
        out.print("GloballyUnified");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, RefCountState state)
{
    switch (state) {
    case EverythingIsLive:
        out.print("EverythingIsLive");
        return;
    case ExactRefCount:
        out.print("ExactRefCount");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, ProofStatus status)
{
    switch (status) {
    case IsProved:
        out.print("IsProved");
        return;
    case NeedsCheck:
        out.print("NeedsCheck");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, CapabilityLevel capabilityLevel)
{
    switch (capabilityLevel) {
    case CannotCompile:
        out.print("CannotCompile");
        return;
    case CanCompile:
        out.print("CanCompile");
        return;
    case CanCompileAndInline:
        out.print("CanCompileAndInline");
        return;
    case CapabilityLevelNotSet:
        out.print("CapabilityLevelNotSet");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

