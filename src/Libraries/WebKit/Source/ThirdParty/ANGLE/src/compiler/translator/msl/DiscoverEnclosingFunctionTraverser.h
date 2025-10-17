/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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

#ifndef COMPILER_TRANSLATOR_MSL_DISCOVERENCLOSINGFUNCTIONTRAVERSER_H_
#define COMPILER_TRANSLATOR_MSL_DISCOVERENCLOSINGFUNCTIONTRAVERSER_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

// A TIntermTraverser that supports discovery of the function a node belongs to.
class DiscoverEnclosingFunctionTraverser : public TIntermTraverser
{
  public:
    DiscoverEnclosingFunctionTraverser(bool preVisit,
                                       bool inVisit,
                                       bool postVisit,
                                       TSymbolTable *symbolTable = nullptr);

    // Returns the function a node belongs inside.
    // Returns null if the node does not belong inside a function.
    const TFunction *discoverEnclosingFunction(TIntermNode *node);
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_MSL_DISCOVERENCLOSINGFUNCTIONTRAVERSER_H_
