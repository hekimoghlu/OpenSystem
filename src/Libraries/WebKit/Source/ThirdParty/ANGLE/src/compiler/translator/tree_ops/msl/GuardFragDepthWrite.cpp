/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GuardFragDepthWrite: Guards use of frag depth behind the function constant
// ANGLEDepthWriteEnabled to ensure it is only used when a valid depth buffer
// is bound.

#include "compiler/translator/tree_ops/msl/GuardFragDepthWrite.h"
#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/msl/AstHelpers.h"
#include "compiler/translator/tree_util/BuiltIn.h"

using namespace sh;

////////////////////////////////////////////////////////////////////////////////

namespace
{

class Rewriter : public TIntermRebuild
{
  public:
    Rewriter(TCompiler &compiler) : TIntermRebuild(compiler, false, true) {}

    PostResult visitBinaryPost(TIntermBinary &node) override
    {
        if (TIntermSymbol *leftSymbolNode = node.getLeft()->getAsSymbolNode())
        {
            if (leftSymbolNode->getType().getQualifier() == TQualifier::EvqFragDepth)
            {
                // This transformation leaves the tree in an inconsistent state by using a variable
                // that's defined in text, outside of the knowledge of the AST.
                // FIXME(jcunningham): remove once function constants (specconst) are implemented
                // with the metal translator.
                mCompiler.disableValidateVariableReferences();

                TSymbolTable *symbolTable = &mCompiler.getSymbolTable();

                // Create kDepthWriteEnabled variable reference.
                TType *boolType = new TType(EbtBool);
                boolType->setQualifier(EvqConst);
                TVariable *depthWriteEnabledVar = new TVariable(
                    symbolTable, sh::ImmutableString(sh::mtl::kDepthWriteEnabledConstName),
                    boolType, SymbolType::AngleInternal);

                TIntermBlock *innerif = new TIntermBlock;
                innerif->appendStatement(&node);

                TIntermSymbol *depthWriteEnabled = new TIntermSymbol(depthWriteEnabledVar);
                TIntermIfElse *ifCall = new TIntermIfElse(depthWriteEnabled, innerif, nullptr);
                return ifCall;
            }
        }

        return node;
    }
};

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool sh::GuardFragDepthWrite(TCompiler &compiler, TIntermBlock &root)
{
    Rewriter rewriter(compiler);
    if (!rewriter.rebuildRoot(root))
    {
        return false;
    }
    return true;
}
