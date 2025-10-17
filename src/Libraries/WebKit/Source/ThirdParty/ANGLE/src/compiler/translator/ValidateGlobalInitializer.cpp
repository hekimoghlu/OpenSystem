/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

#include "compiler/translator/ValidateGlobalInitializer.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

const int kMaxAllowedTraversalDepth = 256;

class ValidateGlobalInitializerTraverser : public TIntermTraverser
{
  public:
    ValidateGlobalInitializerTraverser(int shaderVersion,
                                       bool isWebGL,
                                       bool hasExtNonConstGlobalInitializers);

    void visitSymbol(TIntermSymbol *node) override;
    void visitConstantUnion(TIntermConstantUnion *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitUnary(Visit visit, TIntermUnary *node) override;

    bool isValid() const { return mIsValid && mMaxDepth < mMaxAllowedDepth; }
    bool issueWarning() const { return mIssueWarning; }

  private:
    ANGLE_INLINE void onNonConstInitializerVisit(bool accept)
    {
        if (accept)
        {
            if (!mExtNonConstGlobalInitializers)
            {
                mIssueWarning = true;
            }
        }
        else
        {
            mIsValid = false;
        }
    }

    int mShaderVersion;
    bool mIsWebGL;
    bool mExtNonConstGlobalInitializers;
    bool mIsValid;
    bool mIssueWarning;
};

void ValidateGlobalInitializerTraverser::visitSymbol(TIntermSymbol *node)
{
    // ESSL 1.00 section 4.3 (or ESSL 3.00 section 4.3):
    // Global initializers must be constant expressions.
    switch (node->getType().getQualifier())
    {
        case EvqConst:
            break;
        case EvqGlobal:
        case EvqTemporary:
        case EvqUniform:
            // We allow these cases to be compatible with legacy ESSL 1.00 content.
            // Implement stricter rules for ESSL 3.00 since there's no legacy content to deal
            // with.
            onNonConstInitializerVisit(mExtNonConstGlobalInitializers ||
                                       ((mShaderVersion < 300) && mIsWebGL));
            break;
        default:
            mIsValid = false;
    }
}

void ValidateGlobalInitializerTraverser::visitConstantUnion(TIntermConstantUnion *node)
{
    // Constant unions that are not constant expressions may result from folding a ternary
    // expression.
    switch (node->getType().getQualifier())
    {
        case EvqConst:
            break;
        case EvqTemporary:
            onNonConstInitializerVisit(mExtNonConstGlobalInitializers ||
                                       ((mShaderVersion < 300) && mIsWebGL));
            break;
        default:
            UNREACHABLE();
    }
}

bool ValidateGlobalInitializerTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    // Disallow calls to user-defined functions and texture lookup functions in global variable
    // initializers.  For simplicity, all non-math built-in calls are disallowed.
    if (node->isFunctionCall() ||
        (BuiltInGroup::IsBuiltIn(node->getOp()) && !BuiltInGroup::IsMath(node->getOp())))
    {
        onNonConstInitializerVisit(mExtNonConstGlobalInitializers);
    }
    return true;
}

bool ValidateGlobalInitializerTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (node->isAssignment())
    {
        onNonConstInitializerVisit(mExtNonConstGlobalInitializers);
    }
    return true;
}

bool ValidateGlobalInitializerTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    if (node->isAssignment())
    {
        onNonConstInitializerVisit(mExtNonConstGlobalInitializers);
    }
    return true;
}

ValidateGlobalInitializerTraverser::ValidateGlobalInitializerTraverser(
    int shaderVersion,
    bool isWebGL,
    bool hasExtNonConstGlobalInitializers)
    : TIntermTraverser(true, false, false, nullptr),
      mShaderVersion(shaderVersion),
      mIsWebGL(isWebGL),
      mExtNonConstGlobalInitializers(hasExtNonConstGlobalInitializers),
      mIsValid(true),
      mIssueWarning(false)
{
    setMaxAllowedDepth(kMaxAllowedTraversalDepth);
}

}  // namespace

bool ValidateGlobalInitializer(TIntermTyped *initializer,
                               int shaderVersion,
                               bool isWebGL,
                               bool hasExtNonConstGlobalInitializers,
                               bool *warning)
{
    ValidateGlobalInitializerTraverser validate(shaderVersion, isWebGL,
                                                hasExtNonConstGlobalInitializers);
    initializer->traverse(&validate);
    ASSERT(warning != nullptr);
    *warning = validate.issueWarning();
    return validate.isValid();
}

}  // namespace sh
