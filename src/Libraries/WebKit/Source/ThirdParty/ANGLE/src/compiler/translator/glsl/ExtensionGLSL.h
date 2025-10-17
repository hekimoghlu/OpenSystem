/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
// ExtensionGLSL.h: Defines the TExtensionGLSL class that tracks GLSL extension requirements of
// shaders.

#ifndef COMPILER_TRANSLATOR_GLSL_EXTENSIONGLSL_H_
#define COMPILER_TRANSLATOR_GLSL_EXTENSIONGLSL_H_

#include <set>
#include <string>

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

// Traverses the intermediate tree to determine which GLSL extensions are required
// to support the shader.
class TExtensionGLSL : public TIntermTraverser
{
  public:
    TExtensionGLSL(ShShaderOutput output);

    const std::set<std::string> &getEnabledExtensions() const;
    const std::set<std::string> &getRequiredExtensions() const;

    bool visitUnary(Visit visit, TIntermUnary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

  private:
    void checkOperator(TIntermOperator *node);

    int mTargetVersion;

    std::set<std::string> mEnabledExtensions;
    std::set<std::string> mRequiredExtensions;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_GLSL_EXTENSIONGLSL_H_
