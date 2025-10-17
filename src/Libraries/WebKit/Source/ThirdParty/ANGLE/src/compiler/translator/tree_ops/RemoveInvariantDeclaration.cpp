/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/RemoveInvariantDeclaration.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// An AST traverser that removes invariant declaration for input in fragment shader
// when GLSL >= 4.20 and for output in vertex shader when GLSL < 4.2.
class RemoveInvariantDeclarationTraverser : public TIntermTraverser
{
  public:
    RemoveInvariantDeclarationTraverser() : TIntermTraverser(true, false, false) {}

  private:
    bool visitGlobalQualifierDeclaration(Visit visit,
                                         TIntermGlobalQualifierDeclaration *node) override
    {
        if (node->isInvariant())
        {
            TIntermSequence emptyReplacement;
            mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node,
                                            std::move(emptyReplacement));
        }
        return false;
    }
};

}  // anonymous namespace

bool RemoveInvariantDeclaration(TCompiler *compiler, TIntermNode *root)
{
    RemoveInvariantDeclarationTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
