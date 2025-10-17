/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "compiler/translator/tree_ops/msl/FixTypeConstructors.h"
#include <unordered_map>
#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/msl/AstHelpers.h"

using namespace sh;
////////////////////////////////////////////////////////////////////////////////
namespace
{
class FixTypeTraverser : public TIntermTraverser
{
  public:
    FixTypeTraverser() : TIntermTraverser(false, false, true) {}

    bool visitAggregate(Visit visit, TIntermAggregate *aggregateNode) override
    {
        if (visit != Visit::PostVisit)
        {
            return true;
        }
        if (aggregateNode->isConstructor())
        {
            const TType &retType = aggregateNode->getType();
            if (retType.isScalar())
            {
                // No-op.
            }
            else if (retType.isVector())
            {
                size_t primarySize    = retType.getNominalSize() * retType.getArraySizeProduct();
                TIntermSequence *args = aggregateNode->getSequence();
                size_t argsSize       = 0;
                size_t beforeSize     = 0;
                TIntermNode *lastArg  = nullptr;
                for (TIntermNode *&arg : *args)
                {
                    TIntermTyped *targ = arg->getAsTyped();
                    lastArg            = arg;
                    if (targ)
                    {
                        argsSize += targ->getNominalSize();
                    }
                    if (argsSize <= primarySize)
                    {
                        beforeSize += targ->getNominalSize();
                    }
                }
                if (argsSize > primarySize)
                {
                    size_t swizzleSize         = primarySize - beforeSize;
                    TIntermTyped *targ         = lastArg->getAsTyped();
                    TIntermSwizzle *newSwizzle = nullptr;
                    switch (swizzleSize)
                    {
                        case 1:
                            newSwizzle = new TIntermSwizzle(targ->deepCopy(), {0});
                            break;
                        case 2:
                            newSwizzle = new TIntermSwizzle(targ->deepCopy(), {0, 1});
                            break;
                        case 3:
                            newSwizzle = new TIntermSwizzle(targ->deepCopy(), {0, 1, 2});
                            break;
                        default:
                            UNREACHABLE();  // Should not be reached in case of 0, or 4
                    }
                    if (newSwizzle)
                    {
                        this->queueReplacementWithParent(aggregateNode, lastArg, newSwizzle,
                                                         OriginalNode::IS_DROPPED);
                    }
                }
            }
            else if (retType.isMatrix())
            {
                // TBD if issues
            }
        }
        return true;
    }
};

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool sh::FixTypeConstructors(TCompiler &compiler, SymbolEnv &symbolEnv, TIntermBlock &root)
{
    FixTypeTraverser traverser;
    root.traverse(&traverser);
    if (!traverser.updateTree(&compiler, &root))
    {
        return false;
    }
    return true;
}
