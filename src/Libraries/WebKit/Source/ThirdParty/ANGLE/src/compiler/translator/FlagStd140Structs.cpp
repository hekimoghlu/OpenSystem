/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FlagStd140Structs.cpp: Find structs in std140 blocks, where the padding added in the translator
// conflicts with the "natural" unpadded type.

#include "compiler/translator/FlagStd140Structs.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class FlagStd140StructsTraverser : public TIntermTraverser
{
  public:
    FlagStd140StructsTraverser() : TIntermTraverser(true, false, false) {}

    const std::vector<MappedStruct> getMappedStructs() const { return mMappedStructs; }

  protected:
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;

  private:
    void mapBlockStructMembers(TIntermSymbol *blockDeclarator, const TInterfaceBlock *block);

    std::vector<MappedStruct> mMappedStructs;
};

void FlagStd140StructsTraverser::mapBlockStructMembers(TIntermSymbol *blockDeclarator,
                                                       const TInterfaceBlock *block)
{
    for (auto *field : block->fields())
    {
        if (field->type()->getBasicType() == EbtStruct)
        {
            MappedStruct mappedStruct;
            mappedStruct.blockDeclarator = blockDeclarator;
            mappedStruct.field           = field;
            mMappedStructs.push_back(mappedStruct);
        }
    }
}

bool FlagStd140StructsTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    TIntermTyped *declarator = node->getSequence()->back()->getAsTyped();
    if (declarator->getBasicType() == EbtInterfaceBlock)
    {
        const TInterfaceBlock *block = declarator->getType().getInterfaceBlock();
        if (block->blockStorage() == EbsStd140)
        {
            mapBlockStructMembers(declarator->getAsSymbolNode(), block);
        }
    }
    return false;
}

}  // anonymous namespace

std::vector<MappedStruct> FlagStd140Structs(TIntermNode *node)
{
    FlagStd140StructsTraverser flaggingTraversal;

    node->traverse(&flaggingTraversal);

    return flaggingTraversal.getMappedStructs();
}

}  // namespace sh
