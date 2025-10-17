/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FindSymbol.cpp:
//     Utility for finding a symbol node inside an AST tree.

#include "compiler/translator/tree_util/FindSymbolNode.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class SymbolFinder : public TIntermTraverser
{
  public:
    SymbolFinder(const ImmutableString &symbolName)
        : TIntermTraverser(true, false, false), mSymbolName(symbolName), mNodeFound(nullptr)
    {}

    void visitSymbol(TIntermSymbol *node) override
    {
        if (node->variable().symbolType() != SymbolType::Empty && node->getName() == mSymbolName)
        {
            mNodeFound = node;
        }
    }

    bool isFound() const { return mNodeFound != nullptr; }
    const TIntermSymbol *getNode() const { return mNodeFound; }

  private:
    ImmutableString mSymbolName;
    TIntermSymbol *mNodeFound;
};

}  // anonymous namespace

const TIntermSymbol *FindSymbolNode(TIntermNode *root, const ImmutableString &symbolName)
{
    SymbolFinder finder(symbolName);
    root->traverse(&finder);
    return finder.getNode();
}

}  // namespace sh
