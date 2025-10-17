/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

#ifndef COMPILER_TRANSLATOR_NODETYPE_H_
#define COMPILER_TRANSLATOR_NODETYPE_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

enum class NodeType
{
    Unknown,
    Symbol,
    ConstantUnion,
    FunctionPrototype,
    PreprocessorDirective,
    Unary,
    Binary,
    Ternary,
    Swizzle,
    IfElse,
    Switch,
    Case,
    FunctionDefinition,
    Aggregate,
    Block,
    GlobalQualifierDeclaration,
    Declaration,
    Loop,
    Branch,
};

// This is a function like object instead of a function that stack allocates this because
// TIntermTraverser is a heavy object to construct.
class GetNodeType : private TIntermTraverser
{
  public:
    GetNodeType() : TIntermTraverser(true, false, false) {}

    NodeType operator()(TIntermNode &node)
    {
        node.visit(Visit::PreVisit, this);
        return mNodeType;
    }

  private:
    void visitSymbol(TIntermSymbol *) override { mNodeType = NodeType::Symbol; }

    void visitConstantUnion(TIntermConstantUnion *) override { mNodeType = NodeType::ConstantUnion; }

    void visitFunctionPrototype(TIntermFunctionPrototype *) override
    {
        mNodeType = NodeType::FunctionPrototype;
    }

    void visitPreprocessorDirective(TIntermPreprocessorDirective *) override
    {
        mNodeType = NodeType::PreprocessorDirective;
    }

    bool visitSwizzle(Visit, TIntermSwizzle *) override
    {
        mNodeType = NodeType::Swizzle;
        return false;
    }

    bool visitBinary(Visit, TIntermBinary *) override
    {
        mNodeType = NodeType::Binary;
        return false;
    }

    bool visitUnary(Visit, TIntermUnary *) override
    {
        mNodeType = NodeType::Unary;
        return false;
    }

    bool visitTernary(Visit, TIntermTernary *) override
    {
        mNodeType = NodeType::Ternary;
        return false;
    }

    bool visitIfElse(Visit, TIntermIfElse *) override
    {
        mNodeType = NodeType::IfElse;
        return false;
    }

    bool visitSwitch(Visit, TIntermSwitch *) override
    {
        mNodeType = NodeType::Switch;
        return false;
    }

    bool visitCase(Visit, TIntermCase *) override
    {
        mNodeType = NodeType::Case;
        return false;
    }

    bool visitFunctionDefinition(Visit, TIntermFunctionDefinition *) override
    {
        mNodeType = NodeType::FunctionDefinition;
        return false;
    }

    bool visitAggregate(Visit, TIntermAggregate *) override
    {
        mNodeType = NodeType::Aggregate;
        return false;
    }

    bool visitBlock(Visit, TIntermBlock *) override
    {
        mNodeType = NodeType::Block;
        return false;
    }

    bool visitGlobalQualifierDeclaration(Visit, TIntermGlobalQualifierDeclaration *) override
    {
        mNodeType = NodeType::GlobalQualifierDeclaration;
        return false;
    }

    bool visitDeclaration(Visit, TIntermDeclaration *) override
    {
        mNodeType = NodeType::Declaration;
        return false;
    }

    bool visitLoop(Visit, TIntermLoop *) override
    {
        mNodeType = NodeType::Loop;
        return false;
    }

    bool visitBranch(Visit, TIntermBranch *) override
    {
        mNodeType = NodeType::Branch;
        return false;
    }

    NodeType mNodeType;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_NODETYPE_H_
