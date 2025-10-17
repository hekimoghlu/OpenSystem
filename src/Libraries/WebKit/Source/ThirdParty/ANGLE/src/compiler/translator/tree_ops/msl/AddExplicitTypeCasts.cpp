/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include "compiler/translator/tree_ops/msl/AddExplicitTypeCasts.h"
#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/msl/AstHelpers.h"

using namespace sh;

namespace
{

class Rewriter : public TIntermRebuild
{
    SymbolEnv &mSymbolEnv;
    bool mNeedsExplicitBoolCasts = false;

  public:
    Rewriter(TCompiler &compiler, SymbolEnv &symbolEnv, bool needsExplicitBoolCasts)
        : TIntermRebuild(compiler, false, true),
          mSymbolEnv(symbolEnv),
          mNeedsExplicitBoolCasts(needsExplicitBoolCasts)
    {}

    PostResult visitAggregatePost(TIntermAggregate &callNode) override
    {
        const size_t argCount = callNode.getChildCount();
        const TType &retType  = callNode.getType();

        if (callNode.isConstructor())
        {
            if (IsScalarBasicType(retType))
            {
                if (argCount == 1)
                {
                    TIntermTyped &arg   = GetArg(callNode, 0);
                    const TType argType = arg.getType();
                    if (argType.isVector())
                    {
                        return CoerceSimple(retType, SubVector(arg, 0, 1), mNeedsExplicitBoolCasts);
                    }
                }
            }
            else if (retType.isVector())
            {
                // 1 element arrays need to be accounted for.
                if (argCount == 1 && !retType.isArray())
                {
                    TIntermTyped &arg   = GetArg(callNode, 0);
                    const TType argType = arg.getType();
                    if (argType.isVector())
                    {
                        return CoerceSimple(retType, SubVector(arg, 0, retType.getNominalSize()),
                                            mNeedsExplicitBoolCasts);
                    }
                }
                for (size_t i = 0; i < argCount; ++i)
                {
                    TIntermTyped &arg = GetArg(callNode, i);
                    SetArg(callNode, i,
                           CoerceSimple(retType.getBasicType(), arg, mNeedsExplicitBoolCasts));
                }
            }
            else if (retType.isMatrix())
            {
                if (argCount == 1)
                {
                    TIntermTyped &arg   = GetArg(callNode, 0);
                    const TType argType = arg.getType();
                    if (argType.isMatrix())
                    {
                        if (retType.getCols() != argType.getCols() ||
                            retType.getRows() != argType.getRows())
                        {
                            TemplateArg templateArgs[] = {retType.getCols(), retType.getRows()};
                            return mSymbolEnv.callFunctionOverload(
                                Name("cast"), retType, *new TIntermSequence{&arg}, 2, templateArgs);
                        }
                    }
                }
            }
        }

        return callNode;
    }
};

}  // anonymous namespace

bool sh::AddExplicitTypeCasts(TCompiler &compiler,
                              TIntermBlock &root,
                              SymbolEnv &symbolEnv,
                              bool needsExplicitBoolCasts)
{
    Rewriter rewriter(compiler, symbolEnv, needsExplicitBoolCasts);
    if (!rewriter.rebuildRoot(root))
    {
        return false;
    }
    return true;
}
