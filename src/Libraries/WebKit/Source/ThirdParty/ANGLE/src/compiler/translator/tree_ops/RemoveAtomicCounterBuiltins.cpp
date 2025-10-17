/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveAtomicCounterBuiltins: Remove atomic counter builtins.
//

#include "compiler/translator/tree_ops/RemoveAtomicCounterBuiltins.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{
namespace
{

bool IsAtomicCounterDecl(const TIntermDeclaration *node)
{
    const TIntermSequence &sequence = *(node->getSequence());
    TIntermTyped *variable          = sequence.front()->getAsTyped();
    const TType &type               = variable->getType();
    return type.getQualifier() == EvqUniform && type.isAtomicCounter();
}

// Traverser that removes all GLSL built-ins that use AtomicCounters
// Only called when the builtins are in use, but no atomic counters have been declared
class RemoveAtomicCounterBuiltinsTraverser : public TIntermTraverser
{
  public:
    RemoveAtomicCounterBuiltinsTraverser() : TIntermTraverser(true, false, false) {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        ASSERT(visit == PreVisit);

        // Active atomic counters should have been removed by RewriteAtomicCounters, and this
        // traversal should not have been invoked
        ASSERT(!IsAtomicCounterDecl(node));
        return false;
    }

    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        if (node->getOp() == EOpMemoryBarrierAtomicCounter)
        {
            // Vulkan does not support atomic counters, so if this builtin finds its way here,
            // we need to remove it.
            TIntermSequence emptySequence;
            mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node,
                                            std::move(emptySequence));
            return true;
        }

        // We shouldn't see any other builtins because they cannot be present without an active
        // atomic counter, and should have been removed by RewriteAtomicCounters. If this fires,
        // this traversal should not have been called.
        ASSERT(!(BuiltInGroup::IsBuiltIn(node->getOp()) &&
                 node->getFunction()->isAtomicCounterFunction()));

        return false;
    }
};

}  // anonymous namespace

bool RemoveAtomicCounterBuiltins(TCompiler *compiler, TIntermBlock *root)
{
    RemoveAtomicCounterBuiltinsTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}
}  // namespace sh
