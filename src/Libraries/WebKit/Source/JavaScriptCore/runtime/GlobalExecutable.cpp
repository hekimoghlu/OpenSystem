/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#include "GlobalExecutable.h"

#include "IsoCellSetInlines.h"
#include "JSCellInlines.h"
#include "ScriptExecutableInlines.h"

namespace JSC {

const ClassInfo GlobalExecutable::s_info = { "GlobalExecutable"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(GlobalExecutable) };

template<typename Visitor>
void GlobalExecutable::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* executable = jsCast<GlobalExecutable*>(cell);
    ASSERT_GC_OBJECT_INHERITS(executable, info());
    Base::visitChildren(executable, visitor);
    visitor.append(executable->m_unlinkedCodeBlock);

    if (auto* codeBlock = executable->codeBlock()) {
        // If CodeBlocks is not marked yet, we will run output-constraints.
        // We maintain the invariant that, whenever we see unmarked CodeBlock, then we must run finalizer.
        // And whenever we set a bit on outputConstraintsSet, we must already set a bit in finalizerSet.
        visitCodeBlockEdge(visitor, codeBlock);
        if (!visitor.isMarked(codeBlock)) {
            Heap::ScriptExecutableSpaceAndSets::finalizerSetFor(*executable->subspace()).add(executable);
            Heap::ScriptExecutableSpaceAndSets::outputConstraintsSetFor(*executable->subspace()).add(executable);
        }
    }
}

DEFINE_VISIT_CHILDREN(GlobalExecutable);

template<typename Visitor>
void GlobalExecutable::visitOutputConstraintsImpl(JSCell* cell, Visitor& visitor)
{
    auto* executable = jsCast<GlobalExecutable*>(cell);
    if (CodeBlock* codeBlock = executable->codeBlock()) {
        if (!visitor.isMarked(codeBlock))
            runConstraint(NoLockingNecessary, visitor, codeBlock);
        if (visitor.isMarked(codeBlock))
            Heap::ScriptExecutableSpaceAndSets::outputConstraintsSetFor(*executable->subspace()).remove(executable);
    }
}

DEFINE_VISIT_OUTPUT_CONSTRAINTS(GlobalExecutable);

CodeBlock* GlobalExecutable::replaceCodeBlockWith(VM& vm, CodeBlock* newCodeBlock)
{
    CodeBlock* oldCodeBlock = codeBlock();
    m_codeBlock.setMayBeNull(vm, this, newCodeBlock);
    return oldCodeBlock;
}

void GlobalExecutable::finalizeUnconditionally(VM& vm, CollectionScope)
{
    finalizeCodeBlockEdge(vm, m_codeBlock);
    Heap::ScriptExecutableSpaceAndSets::outputConstraintsSetFor(*subspace()).remove(this);
    Heap::ScriptExecutableSpaceAndSets::finalizerSetFor(*subspace()).remove(this);
}

} // namespace JSC
