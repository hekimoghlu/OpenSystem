/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

#include "CodeBlock.h"
#include "ScriptExecutable.h"

namespace JSC {

inline void ScriptExecutable::finalizeCodeBlockEdge(VM& vm, WriteBarrier<CodeBlock>& codeBlockEdge)
{
    for (;;) {
        auto* codeBlock = codeBlockEdge.get();
        if (!codeBlock)
            return;

        if (vm.heap.isMarked(codeBlock))
            return;

        if (codeBlock->shouldJettisonDueToWeakReference(vm))
            codeBlock->jettison(Profiler::JettisonDueToWeakReference);
        else
            codeBlock->jettison(Profiler::JettisonDueToOldAge);

        if (codeBlock == codeBlockEdge.get()) {
            codeBlockEdge.clear();
            return;
        }
    }
}

} // namespace JSC

