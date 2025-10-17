/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Defines analyses of the AST needed for HLSL generation

#ifndef COMPILER_TRANSLATOR_HLSL_ASTMETADATAHLSL_H_
#define COMPILER_TRANSLATOR_HLSL_ASTMETADATAHLSL_H_

#include <set>
#include <vector>

namespace sh
{

class CallDAG;
class TIntermNode;
class TIntermIfElse;
class TIntermLoop;

struct ASTMetadataHLSL
{
    ASTMetadataHLSL()
        : mUsesGradient(false),
          mCalledInDiscontinuousLoop(false),
          mHasGradientLoopInCallGraph(false),
          mNeedsLod0(false)
    {}

    // Here "something uses a gradient" means here that it either contains a
    // gradient operation, or a call to a function that uses a gradient.
    bool hasGradientInCallGraph(TIntermLoop *node);
    bool hasGradientLoop(TIntermIfElse *node);

    // Does the function use a gradient.
    bool mUsesGradient;

    // Even if usesGradient is true, some control flow might not use a gradient
    // so we store the set of all gradient-using control flows.
    std::set<TIntermNode *> mControlFlowsContainingGradient;

    // Remember information about the discontinuous loops and which functions
    // are called in such loops.
    bool mCalledInDiscontinuousLoop;
    bool mHasGradientLoopInCallGraph;
    std::set<TIntermLoop *> mDiscontinuousLoops;
    std::set<TIntermIfElse *> mIfsContainingGradientLoop;

    // Will we need to generate a Lod0 version of the function.
    bool mNeedsLod0;
};

typedef std::vector<ASTMetadataHLSL> MetadataList;

// Return the AST analysis result, in the order defined by the call DAG
MetadataList CreateASTMetadataHLSL(TIntermNode *root, const CallDAG &callDag);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HLSL_ASTMETADATAHLSL_H_
