/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ValidateMaxParameters checks if function definitions have more than a set number of parameters.

#include "compiler/translator/ValidateMaxParameters.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

bool ValidateMaxParameters(TIntermBlock *root, unsigned int maxParameters)
{
    for (TIntermNode *node : *root->getSequence())
    {
        TIntermFunctionDefinition *definition = node->getAsFunctionDefinition();
        if (definition != nullptr &&
            definition->getFunctionPrototype()->getFunction()->getParamCount() > maxParameters)
        {
            return false;
        }
    }
    return true;
}

}  // namespace sh
