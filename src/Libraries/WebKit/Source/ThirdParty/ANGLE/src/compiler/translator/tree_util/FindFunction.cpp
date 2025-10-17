/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindFunction.cpp: Find functions.

#include "compiler/translator/tree_util/FindFunction.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

size_t FindFirstFunctionDefinitionIndex(TIntermBlock *root)
{
    const TIntermSequence &sequence = *root->getSequence();
    for (size_t index = 0; index < sequence.size(); ++index)
    {
        TIntermNode *node                       = sequence[index];
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr)
        {
            return index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

}  // namespace sh
