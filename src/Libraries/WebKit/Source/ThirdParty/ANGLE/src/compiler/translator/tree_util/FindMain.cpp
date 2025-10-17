/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindMain.cpp: Find the main() function definition in a given AST.

#include "compiler/translator/tree_util/FindMain.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

size_t FindMainIndex(TIntermBlock *root)
{
    const TIntermSequence &sequence = *root->getSequence();
    for (size_t index = 0; index < sequence.size(); ++index)
    {
        TIntermNode *node                       = sequence[index];
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr && nodeFunction->getFunction()->isMain())
        {
            return index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

TIntermFunctionDefinition *FindMain(TIntermBlock *root)
{
    for (TIntermNode *node : *root->getSequence())
    {
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr && nodeFunction->getFunction()->isMain())
        {
            return nodeFunction;
        }
    }
    return nullptr;
}

TIntermBlock *FindMainBody(TIntermBlock *root)
{
    TIntermFunctionDefinition *main = FindMain(root);
    ASSERT(main != nullptr);
    TIntermBlock *mainBody = main->getBody();
    ASSERT(mainBody != nullptr);
    return mainBody;
}

}  // namespace sh
