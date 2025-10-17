/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "ProfilerJettisonReason.h"

#include <wtf/PrintStream.h>

namespace WTF {

using namespace JSC::Profiler;

void printInternal(PrintStream& out, JettisonReason reason)
{
    switch (reason) {
    case NotJettisoned:
        out.print("NotJettisoned");
        return;
    case JettisonDueToWeakReference:
        out.print("WeakReference");
        return;
    case JettisonDueToDebuggerBreakpoint:
        out.print("DebuggerBreakpoint");
        return;
    case JettisonDueToDebuggerStepping:
        out.print("DebuggerStepping");
        return;
    case JettisonDueToBaselineLoopReoptimizationTrigger:
        out.print("BaselineLoopReoptimizationTrigger");
        return;
    case JettisonDueToBaselineLoopReoptimizationTriggerOnOSREntryFail:
        out.print("BaselineLoopReoptimizationTriggerOnOSREntryFail");
        return;
    case JettisonDueToOSRExit:
        out.print("OSRExit");
        return;
    case JettisonDueToProfiledWatchpoint:
        out.print("ProfiledWatchpoint");
        return;
    case JettisonDueToUnprofiledWatchpoint:
        out.print("UnprofiledWatchpoint");
        return;
    case JettisonDueToOldAge:
        out.print("JettisonDueToOldAge");
        return;
    case JettisonDueToVMTraps:
        out.print("JettisonDueToVMTraps");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

