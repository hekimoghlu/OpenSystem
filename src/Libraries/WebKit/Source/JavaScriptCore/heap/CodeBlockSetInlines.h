/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "CodeBlockSet.h"

namespace JSC {

inline void CodeBlockSet::mark(const AbstractLocker&, CodeBlock* codeBlock)
{
    if (!codeBlock)
        return;

    // Conservative root scanning in Eden collection can only find PreciseAllocation that is allocated in this Eden cycle.
    // Since CodeBlockSet::m_currentlyExecuting is strongly assuming that this catches all the currently executing CodeBlock,
    // we now have a restriction that all CodeBlock needs to be a non-precise-allocation.
    ASSERT(!codeBlock->isPreciseAllocation());
    m_currentlyExecuting.add(codeBlock);
}

template<typename Functor>
void CodeBlockSet::iterate(const Functor& functor)
{
    Locker locker { m_lock };
    iterate(locker, functor);
}

template<typename Functor>
void CodeBlockSet::iterate(const AbstractLocker&, const Functor& functor)
{
    for (CodeBlock* codeBlock : m_codeBlocks)
        functor(codeBlock);
}

template<typename Functor>
void CodeBlockSet::iterateCurrentlyExecuting(const Functor& functor)
{
    Locker locker { m_lock };
    for (CodeBlock* codeBlock : m_currentlyExecuting)
        functor(codeBlock);
}

} // namespace JSC

