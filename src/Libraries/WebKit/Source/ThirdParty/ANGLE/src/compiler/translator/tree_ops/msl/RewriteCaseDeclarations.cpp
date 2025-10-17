/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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

#include "compiler/translator/tree_ops/msl/RewriteCaseDeclarations.h"
#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/util.h"

using namespace sh;

////////////////////////////////////////////////////////////////////////////////

namespace
{

class Rewriter : public TIntermRebuild
{
    std::vector<std::vector<const TVariable *>> mDeclaredVarStack;

  public:
    Rewriter(TCompiler &compiler) : TIntermRebuild(compiler, true, true) {}

    ~Rewriter() override { ASSERT(mDeclaredVarStack.empty()); }

  private:
    PreResult visitSwitchPre(TIntermSwitch &node) override
    {
        mDeclaredVarStack.emplace_back();
        return node;
    }

    PostResult visitSwitchPost(TIntermSwitch &node) override
    {
        ASSERT(!mDeclaredVarStack.empty());
        const auto vars = std::move(mDeclaredVarStack.back());
        mDeclaredVarStack.pop_back();
        if (!vars.empty())
        {
            auto &block = *new TIntermBlock();
            for (const TVariable *var : vars)
            {
                block.appendStatement(new TIntermDeclaration{var});
            }
            block.appendStatement(&node);
            return block;
        }
        return node;
    }

    PreResult visitDeclarationPre(TIntermDeclaration &node) override
    {
        if (!mDeclaredVarStack.empty())
        {
            TIntermNode *parent = getParentNode();
            if (parent->getAsBlock())
            {
                TIntermNode *grandparent = getParentNode(1);
                if (grandparent && grandparent->getAsSwitchNode())
                {
                    Declaration decl = ViewDeclaration(node);
                    mDeclaredVarStack.back().push_back(&decl.symbol.variable());
                    if (decl.initExpr)
                    {
                        return *new TIntermBinary(TOperator::EOpAssign, &decl.symbol,
                                                  decl.initExpr);
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
        }
        return node;
    }
};

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool sh::RewriteCaseDeclarations(TCompiler &compiler, TIntermBlock &root)
{
    if (!Rewriter(compiler).rebuildRoot(root))
    {
        return false;
    }
    return true;
}
