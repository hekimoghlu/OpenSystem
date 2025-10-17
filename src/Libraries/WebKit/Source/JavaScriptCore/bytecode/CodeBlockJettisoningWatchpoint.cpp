/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "CodeBlockJettisoningWatchpoint.h"

#include "CodeBlockInlines.h"
#include "DFGCommon.h"

namespace JSC {

void CodeBlockJettisoningWatchpoint::fireInternal(VM&, const FireDetail& detail)
{
    ASSERT(!m_owner->wasDestructed());
    // If CodeBlock is no longer live, we do not fire it.
    // This works since CodeBlock is the owner of this watchpoint. When it gets destroyed, then this watchpoint also gets destroyed.
    // Only problematic case is, (1) CodeBlock is dead, but (2) destructor is not called yet.
    if (m_owner->isPendingDestruction())
        return;

    if (DFG::shouldDumpDisassembly())
        dataLog("Firing watchpoint ", RawPointer(this), " on ", *m_owner, "\n");

    m_owner->jettison(Profiler::JettisonDueToUnprofiledWatchpoint, CountReoptimization, &detail);
}

} // namespace JSC

