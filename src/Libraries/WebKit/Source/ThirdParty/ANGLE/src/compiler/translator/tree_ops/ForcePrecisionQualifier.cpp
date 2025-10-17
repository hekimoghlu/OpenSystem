/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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

#include "compiler/translator/tree_ops/ForcePrecisionQualifier.h"
#include "angle_gl.h"
#include "common/debug.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{
class TPrecisionTraverser : public TIntermTraverser
{
  public:
    TPrecisionTraverser(TSymbolTable *symbolTable);

  protected:
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;

    void overwriteVariablePrecision(TType *type) const;
};

TPrecisionTraverser::TPrecisionTraverser(TSymbolTable *symbolTable)
    : TIntermTraverser(true, true, true, symbolTable)
{}

void TPrecisionTraverser::overwriteVariablePrecision(TType *type) const
{
    if (type->getPrecision() == EbpHigh)
    {
        type->setPrecision(EbpMedium);
    }
}

bool TPrecisionTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    // Variable declaration.
    if (visit == PreVisit)
    {
        const TIntermSequence &sequence = *(node->getSequence());
        TIntermTyped *variable          = sequence.front()->getAsTyped();
        const TType &type               = variable->getType();
        TQualifier qualifier            = variable->getQualifier();

        // Don't modify uniform since it might be shared between vertex and fragment shader
        if (qualifier == EvqUniform)
        {
            return true;
        }

        // Visit the struct.
        if (type.isStructSpecifier())
        {
            const TStructure *structure = type.getStruct();
            const TFieldList &fields    = structure->fields();
            for (size_t i = 0; i < fields.size(); ++i)
            {
                const TField *field    = fields[i];
                const TType *fieldType = field->type();
                overwriteVariablePrecision((TType *)fieldType);
            }
        }
        else if (type.getBasicType() == EbtInterfaceBlock)
        {
            const TInterfaceBlock *interfaceBlock = type.getInterfaceBlock();
            const TFieldList &fields              = interfaceBlock->fields();
            for (const TField *field : fields)
            {
                const TType *fieldType = field->type();
                overwriteVariablePrecision((TType *)fieldType);
            }
        }
        else
        {
            overwriteVariablePrecision((TType *)&type);
        }
    }
    return true;
}
}  // namespace

bool ForceShaderPrecisionToMediump(TIntermNode *root, TSymbolTable *symbolTable, GLenum shaderType)
{
    if (shaderType != GL_FRAGMENT_SHADER)
    {
        return true;
    }

    TPrecisionTraverser traverser(symbolTable);
    root->traverse(&traverser);
    return true;
}

}  // namespace sh
