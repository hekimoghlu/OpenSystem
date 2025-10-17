/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "compiler/translator/msl/DiscoverDependentFunctions.h"
#include "compiler/translator/msl/DiscoverEnclosingFunctionTraverser.h"
#include "compiler/translator/msl/MapFunctionsToDefinitions.h"

using namespace sh;

////////////////////////////////////////////////////////////////////////////////

namespace
{

class Discoverer : public DiscoverEnclosingFunctionTraverser
{
  private:
    const std::function<bool(const TVariable &)> &mVars;
    const FunctionToDefinition &mFuncToDef;
    std::unordered_set<const TFunction *> mNonDepFunctions;

  public:
    std::unordered_set<const TFunction *> mDepFunctions;

  public:
    Discoverer(const std::function<bool(const TVariable &)> &vars,
               const FunctionToDefinition &funcToDef)
        : DiscoverEnclosingFunctionTraverser(true, false, true), mVars(vars), mFuncToDef(funcToDef)
    {}

    void visitSymbol(TIntermSymbol *symbolNode) override
    {
        const TVariable &var = symbolNode->variable();
        if (!mVars(var))
        {
            return;
        }
        const TFunction *owner = discoverEnclosingFunction(symbolNode);
        if (owner)
        {
            mDepFunctions.insert(owner);
        }
    }

    bool visitAggregate(Visit visit, TIntermAggregate *aggregateNode) override
    {
        if (visit != Visit::PreVisit)
        {
            return true;
        }

        if (!aggregateNode->isConstructor())
        {
            const TFunction *func = aggregateNode->getFunction();

            if (mNonDepFunctions.find(func) != mNonDepFunctions.end())
            {
                return true;
            }

            if (mDepFunctions.find(func) == mDepFunctions.end())
            {
                auto it = mFuncToDef.find(func);
                if (it == mFuncToDef.end())
                {
                    return true;
                }

                // Recursion is banned in GLSL, so I believe AngleIR has this property too.
                // This implementation assumes (direct and mutual) recursion is prohibited.
                TIntermFunctionDefinition &funcDefNode = *it->second;
                funcDefNode.traverse(this);
                if (mNonDepFunctions.find(func) != mNonDepFunctions.end())
                {
                    return true;
                }
                ASSERT(mDepFunctions.find(func) != mDepFunctions.end());
            }

            const TFunction *owner = discoverEnclosingFunction(aggregateNode);
            ASSERT(owner);
            mDepFunctions.insert(owner);
        }

        return true;
    }

    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *funcDefNode) override
    {
        const TFunction *func = funcDefNode->getFunction();

        if (visit != Visit::PostVisit)
        {
            if (mDepFunctions.find(func) != mDepFunctions.end())
            {
                return false;
            }

            if (mNonDepFunctions.find(func) != mNonDepFunctions.end())
            {
                return false;
            }

            return true;
        }

        if (mDepFunctions.find(func) == mDepFunctions.end())
        {
            mNonDepFunctions.insert(func);
        }

        return true;
    }
};

}  // namespace

std::unordered_set<const TFunction *> sh::DiscoverDependentFunctions(
    TIntermBlock &root,
    const std::function<bool(const TVariable &)> &vars)
{
    const FunctionToDefinition funcToDef = MapFunctionsToDefinitions(root);
    Discoverer discoverer(vars, funcToDef);
    root.traverse(&discoverer);
    return std::move(discoverer.mDepFunctions);
}
