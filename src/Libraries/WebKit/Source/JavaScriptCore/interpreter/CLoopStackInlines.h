/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

#if ENABLE(C_LOOP)

#include "CLoopStack.h"
#include "CallFrame.h"
#include "CodeBlock.h"
#include "VM.h"

namespace JSC {

inline bool CLoopStack::ensureCapacityFor(Register* newTopOfStack)
{
    if (newTopOfStack >= m_end)
        return true;
    return grow(newTopOfStack);
}

inline void* CLoopStack::currentStackPointer() const
{
    // One might be tempted to assert that m_currentStackPointer <= m_topCallFrame->topOfFrame()
    // here. That assertion would be incorrect because this function may be called from function
    // prologues (e.g. during a stack check) where m_currentStackPointer may be higher than
    // m_topCallFrame->topOfFrame() because the stack pointer has not been initialized to point
    // to frame top yet.
    return m_currentStackPointer;
}

inline void CLoopStack::setCLoopStackLimit(Register* newTopOfStack)
{
    m_end = newTopOfStack;
    m_vm.setCLoopStackLimit(newTopOfStack);
}

} // namespace JSC

#endif // ENABLE(C_LOOP)
