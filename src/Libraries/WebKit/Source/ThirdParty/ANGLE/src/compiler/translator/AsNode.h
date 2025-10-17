/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#ifndef COMPILER_TRANSLATOR_ASNODE_H_
#define COMPILER_TRANSLATOR_ASNODE_H_

#include "common/angleutils.h"
#include "compiler/translator/IntermNode.h"

#include <utility>

namespace sh
{

namespace priv
{

template <typename T>
struct AsNode
{};

template <>
struct AsNode<TIntermNode>
{
    static ANGLE_INLINE TIntermNode *exec(TIntermNode *node) { return node; }
};

template <>
struct AsNode<TIntermTyped>
{
    static ANGLE_INLINE TIntermTyped *exec(TIntermNode *node)
    {
        return node ? node->getAsTyped() : nullptr;
    }
};

template <>
struct AsNode<TIntermSymbol>
{
    static ANGLE_INLINE TIntermSymbol *exec(TIntermNode *node)
    {
        return node ? node->getAsSymbolNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermConstantUnion>
{
    static ANGLE_INLINE TIntermConstantUnion *exec(TIntermNode *node)
    {
        return node ? node->getAsConstantUnion() : nullptr;
    }
};

template <>
struct AsNode<TIntermFunctionPrototype>
{
    static ANGLE_INLINE TIntermFunctionPrototype *exec(TIntermNode *node)
    {
        return node ? node->getAsFunctionPrototypeNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermPreprocessorDirective>
{
    static ANGLE_INLINE TIntermPreprocessorDirective *exec(TIntermNode *node)
    {
        return node ? node->getAsPreprocessorDirective() : nullptr;
    }
};

template <>
struct AsNode<TIntermSwizzle>
{
    static ANGLE_INLINE TIntermSwizzle *exec(TIntermNode *node)
    {
        return node ? node->getAsSwizzleNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermBinary>
{
    static ANGLE_INLINE TIntermBinary *exec(TIntermNode *node)
    {
        return node ? node->getAsBinaryNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermUnary>
{
    static ANGLE_INLINE TIntermUnary *exec(TIntermNode *node)
    {
        return node ? node->getAsUnaryNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermTernary>
{
    static ANGLE_INLINE TIntermTernary *exec(TIntermNode *node)
    {
        return node ? node->getAsTernaryNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermIfElse>
{
    static ANGLE_INLINE TIntermIfElse *exec(TIntermNode *node)
    {
        return node ? node->getAsIfElseNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermSwitch>
{
    static ANGLE_INLINE TIntermSwitch *exec(TIntermNode *node)
    {
        return node ? node->getAsSwitchNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermCase>
{
    static ANGLE_INLINE TIntermCase *exec(TIntermNode *node)
    {
        return node ? node->getAsCaseNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermFunctionDefinition>
{
    static ANGLE_INLINE TIntermFunctionDefinition *exec(TIntermNode *node)
    {
        return node ? node->getAsFunctionDefinition() : nullptr;
    }
};

template <>
struct AsNode<TIntermAggregate>
{
    static ANGLE_INLINE TIntermAggregate *exec(TIntermNode *node)
    {
        return node ? node->getAsAggregate() : nullptr;
    }
};

template <>
struct AsNode<TIntermBlock>
{
    static ANGLE_INLINE TIntermBlock *exec(TIntermNode *node)
    {
        return node ? node->getAsBlock() : nullptr;
    }
};

template <>
struct AsNode<TIntermGlobalQualifierDeclaration>
{
    static ANGLE_INLINE TIntermGlobalQualifierDeclaration *exec(TIntermNode *node)
    {
        return node ? node->getAsGlobalQualifierDeclarationNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermDeclaration>
{
    static ANGLE_INLINE TIntermDeclaration *exec(TIntermNode *node)
    {
        return node ? node->getAsDeclarationNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermLoop>
{
    static ANGLE_INLINE TIntermLoop *exec(TIntermNode *node)
    {
        return node ? node->getAsLoopNode() : nullptr;
    }
};

template <>
struct AsNode<TIntermBranch>
{
    static ANGLE_INLINE TIntermBranch *exec(TIntermNode *node)
    {
        return node ? node->getAsBranchNode() : nullptr;
    }
};

}  // namespace priv

template <typename T>
ANGLE_INLINE T *asNode(TIntermNode *node)
{
    return priv::AsNode<T>::exec(node);
}

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_ASNODE_H_
