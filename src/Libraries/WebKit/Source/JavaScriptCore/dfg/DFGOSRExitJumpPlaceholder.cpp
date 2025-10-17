/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "DFGOSRExitJumpPlaceholder.h"

#if ENABLE(DFG_JIT)

#include "DFGJITCompiler.h"
#include "DFGSpeculativeJIT.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void OSRExitJumpPlaceholder::fill(JITCompiler& jit, const MacroAssembler::JumpList& jumps)
{
    if (!*this)
        return;
    jit.m_exitCompilationInfo[m_index].m_failureJumps = jumps;
}

void OSRExitJumpPlaceholder::fill(SpeculativeJIT* jit, const MacroAssembler::JumpList& jumps)
{
    fill(*jit, jumps);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

