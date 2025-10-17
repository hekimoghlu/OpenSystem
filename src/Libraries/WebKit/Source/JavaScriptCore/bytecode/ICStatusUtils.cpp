/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#include "ICStatusUtils.h"

#include "CodeBlock.h"
#include "ExitFlag.h"
#include "UnlinkedCodeBlock.h"

namespace JSC {

ExitFlag hasBadCacheExitSite(CodeBlock* profiledBlock, BytecodeIndex bytecodeIndex)
{
#if ENABLE(DFG_JIT)
    UnlinkedCodeBlock* unlinkedCodeBlock = profiledBlock->unlinkedCodeBlock();
    ConcurrentJSLocker locker(unlinkedCodeBlock->m_lock);
    auto exitFlag = [&] (ExitKind exitKind) -> ExitFlag {
        auto withInlined = [&] (ExitingInlineKind inlineKind) -> ExitFlag {
            return ExitFlag(unlinkedCodeBlock->hasExitSite(locker, DFG::FrequentExitSite(bytecodeIndex, exitKind, ExitFromAnything, inlineKind)), inlineKind);
        };
        return withInlined(ExitFromNotInlined) | withInlined(ExitFromInlined);
    };
    return exitFlag(BadCache) | exitFlag(BadConstantCache) | exitFlag(BadType);
#else
    UNUSED_PARAM(profiledBlock);
    UNUSED_PARAM(bytecodeIndex);
    return ExitFlag();
#endif
}

} // namespace JSC

