/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
// FlagSamplersForTexelFetch.cpp: finds all instances of texelFetch used with a static reference to
// a sampler uniform, and flag that uniform as having been used with texelFetch
//

#include "compiler/translator/tree_ops/spirv/FlagSamplersWithTexelFetch.h"

#include "angle_gl.h"
#include "common/utilities.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"

namespace sh
{
namespace
{

class FlagSamplersWithTexelFetchTraverser : public TIntermTraverser
{
  public:
    FlagSamplersWithTexelFetchTraverser(TSymbolTable *symbolTable,
                                        std::vector<ShaderVariable> *uniforms)
        : TIntermTraverser(true, true, true, symbolTable), mUniforms(uniforms)
    {}

    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        // Decide if the node is a call to texelFetch[Offset]
        if (!BuiltInGroup::IsBuiltIn(node->getOp()))
        {
            return true;
        }

        ASSERT(node->getFunction()->symbolType() == SymbolType::BuiltIn);
        if (node->getFunction()->name() != "texelFetch" &&
            node->getFunction()->name() != "texelFetchOffset")
        {
            return true;
        }

        const TIntermSequence *sequence = node->getSequence();

        ASSERT(sequence->size() > 0);

        TIntermSymbol *samplerSymbol = sequence->at(0)->getAsSymbolNode();
        ASSERT(samplerSymbol != nullptr);

        const TVariable &samplerVariable = samplerSymbol->variable();

        for (ShaderVariable &uniform : *mUniforms)
        {
            if (samplerVariable.name() == uniform.name)
            {
                ASSERT(gl::IsSamplerType(uniform.type));
                uniform.texelFetchStaticUse = true;
                break;
            }
        }

        return true;
    }

  private:
    std::vector<ShaderVariable> *mUniforms;
};

}  // anonymous namespace

bool FlagSamplersForTexelFetch(TCompiler *compiler,
                               TIntermBlock *root,
                               TSymbolTable *symbolTable,
                               std::vector<ShaderVariable> *uniforms)
{
    ASSERT(uniforms != nullptr);
    if (uniforms->size() > 0)
    {
        FlagSamplersWithTexelFetchTraverser traverser(symbolTable, uniforms);
        root->traverse(&traverser);
    }

    return true;
}

}  // namespace sh
