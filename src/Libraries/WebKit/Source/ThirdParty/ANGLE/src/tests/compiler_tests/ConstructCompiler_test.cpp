/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
// ConstructCompiler_test.cpp
//   Test the sh::ConstructCompiler interface with different parameters.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"

// Test default parameters.
TEST(ConstructCompilerTest, DefaultParameters)
{
    ShBuiltInResources resources;
    sh::InitBuiltInResources(&resources);
    ShHandle compiler = sh::ConstructCompiler(GL_FRAGMENT_SHADER, SH_WEBGL_SPEC,
                                              SH_GLSL_COMPATIBILITY_OUTPUT, &resources);
    ASSERT_NE(nullptr, compiler);
    sh::Destruct(compiler);
}

// Test invalid MaxDrawBuffers zero.
TEST(ConstructCompilerTest, InvalidMaxDrawBuffers)
{
    ShBuiltInResources resources;
    sh::InitBuiltInResources(&resources);
    resources.MaxDrawBuffers = 0;
    ShHandle compiler        = sh::ConstructCompiler(GL_FRAGMENT_SHADER, SH_WEBGL_SPEC,
                                              SH_GLSL_COMPATIBILITY_OUTPUT, &resources);
    ASSERT_EQ(nullptr, compiler);
}

// Test invalid MaxDualSourceDrawBuffers zero.
TEST(ConstructCompilerTest, InvalidMaxDualSourceDrawBuffers)
{
    ShBuiltInResources resources;
    sh::InitBuiltInResources(&resources);
    resources.EXT_blend_func_extended  = 1;
    resources.MaxDualSourceDrawBuffers = 0;
    ShHandle compiler                  = sh::ConstructCompiler(GL_FRAGMENT_SHADER, SH_WEBGL_SPEC,
                                              SH_GLSL_COMPATIBILITY_OUTPUT, &resources);
    ASSERT_EQ(nullptr, compiler);
}
