/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "compiler/translator/tree_ops/msl/IntroduceVertexIndexID.h"
#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/msl/AstHelpers.h"
#include "compiler/translator/tree_util/BuiltIn.h"
using namespace sh;

////////////////////////////////////////////////////////////////////////////////

namespace
{

constexpr const TVariable kgl_VertexIDMetal(BuiltInId::gl_VertexID,
                                            ImmutableString("vertexIDMetal"),
                                            SymbolType::AngleInternal,
                                            TExtension::UNDEFINED,
                                            StaticType::Get<EbtUInt, EbpHigh, EvqVertexID, 1, 1>());

constexpr const TVariable kgl_instanceIdMetal(
    BuiltInId::gl_InstanceID,
    ImmutableString("instanceIdMod"),
    SymbolType::AngleInternal,
    TExtension::UNDEFINED,
    StaticType::Get<EbtUInt, EbpHigh, EvqInstanceID, 1, 1>());

constexpr const TVariable kgl_baseInstanceMetal(
    BuiltInId::gl_BaseInstance,
    ImmutableString("baseInstance"),
    SymbolType::AngleInternal,
    TExtension::UNDEFINED,
    StaticType::Get<EbtUInt, EbpHigh, EvqInstanceID, 1, 1>());

class Rewriter : public TIntermRebuild
{
  public:
    Rewriter(TCompiler &compiler) : TIntermRebuild(compiler, true, true) {}

  private:
    PreResult visitFunctionDefinitionPre(TIntermFunctionDefinition &node) override
    {
        if (node.getFunction()->isMain())
        {
            const TFunction *mainFunction = node.getFunction();
            bool needsVertexId            = true;
            bool needsInstanceId          = true;
            std::vector<const TVariable *> mVariablesToIntroduce;
            for (size_t i = 0; i < mainFunction->getParamCount(); ++i)
            {
                const TVariable *param = mainFunction->getParam(i);
                Name instanceIDName =
                    Pipeline{Pipeline::Type::InstanceId, nullptr}.getStructInstanceName(
                        Pipeline::Variant::Modified);
                if (Name(*param) == instanceIDName)
                {
                    needsInstanceId = false;
                }
                else if (param->getType().getQualifier() == TQualifier::EvqVertexID)
                {
                    needsVertexId = false;
                }
            }
            if (needsInstanceId)
            {
                // Ensure these variables are present because they are required for XFB emulation.
                mVariablesToIntroduce.push_back(&kgl_instanceIdMetal);
                mVariablesToIntroduce.push_back(&kgl_baseInstanceMetal);
            }
            if (needsVertexId)
            {
                mVariablesToIntroduce.push_back(&kgl_VertexIDMetal);
            }
            const TFunction &newFunction = CloneFunctionAndAppendParams(
                mSymbolTable, nullptr, *node.getFunction(), mVariablesToIntroduce);
            TIntermFunctionPrototype *newProto = new TIntermFunctionPrototype(&newFunction);
            return new TIntermFunctionDefinition(newProto, node.getBody());
        }
        return node;
    }
};

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool sh::IntroduceVertexAndInstanceIndex(TCompiler &compiler, TIntermBlock &root)
{
    if (!Rewriter(compiler).rebuildRoot(root))
    {
        return false;
    }
    return true;
}
