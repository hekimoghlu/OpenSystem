/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#include "CodeBlockSet.h"

#include "CodeBlock.h"
#include "HeapInlines.h"
#include <wtf/CommaPrinter.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CodeBlockSet);

CodeBlockSet::CodeBlockSet() = default;

CodeBlockSet::~CodeBlockSet() = default;

bool CodeBlockSet::contains(const AbstractLocker&, void* candidateCodeBlock)
{
    RELEASE_ASSERT(m_lock.isLocked());
    CodeBlock* codeBlock = static_cast<CodeBlock*>(candidateCodeBlock);
    if (!UncheckedKeyHashSet<CodeBlock*>::isValidValue(codeBlock))
        return false;
    return m_codeBlocks.contains(codeBlock);
}

void CodeBlockSet::clearCurrentlyExecutingAndRemoveDeadCodeBlocks(VM& vm)
{
    ASSERT(vm.heap.isInPhase(CollectorPhase::End));
    m_currentlyExecuting.clear();
    m_codeBlocks.removeIf([&](CodeBlock* codeBlock) {
        return !vm.heap.isMarked(codeBlock);
    });
}

bool CodeBlockSet::isCurrentlyExecuting(CodeBlock* codeBlock)
{
    return m_currentlyExecuting.contains(codeBlock);
}

void CodeBlockSet::dump(PrintStream& out) const
{
    CommaPrinter comma;
    out.print("{codeBlocks = ["_s);
    for (CodeBlock* codeBlock : m_codeBlocks)
        out.print(comma, pointerDump(codeBlock));
    out.print("], currentlyExecuting = ["_s);
    comma = CommaPrinter();
    for (CodeBlock* codeBlock : m_currentlyExecuting)
        out.print(comma, pointerDump(codeBlock));
    out.print("]}"_s);
}

void CodeBlockSet::add(CodeBlock* codeBlock)
{
    Locker locker { m_lock };
    auto result = m_codeBlocks.add(codeBlock);
    RELEASE_ASSERT(result);
}

void CodeBlockSet::remove(CodeBlock* codeBlock)
{
    Locker locker { m_lock };
    bool result = m_codeBlocks.remove(codeBlock);
    RELEASE_ASSERT(result);
}

} // namespace JSC

