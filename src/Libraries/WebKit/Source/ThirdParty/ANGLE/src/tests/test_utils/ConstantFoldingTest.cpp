/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
// ConstantFoldingTest.cpp:
//   Utilities for constant folding tests.
//

#include "tests/test_utils/ConstantFoldingTest.h"

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "compiler/translator/glsl/TranslatorESSL.h"

using namespace sh;

void ConstantFoldingExpressionTest::evaluate(const std::string &type, const std::string &expression)
{
    // We first assign the expression into a const variable so we can also verify that it gets
    // qualified as a constant expression. We then assign that constant expression into my_FragColor
    // to make sure that the value is not pruned.
    std::stringstream shaderStream;
    shaderStream << "#version 310 es\n"
                    "precision mediump float;\n"
                 << "out " << type << " my_FragColor;\n"
                 << "void main()\n"
                    "{\n"
                 << "    const " << type << " v = " << expression << ";\n"
                 << "    my_FragColor = v;\n"
                    "}\n";
    compileAssumeSuccess(shaderStream.str());
}

void ConstantFoldingExpressionTest::evaluateIvec4(const std::string &ivec4Expression)
{
    evaluate("ivec4", ivec4Expression);
}

void ConstantFoldingExpressionTest::evaluateVec4(const std::string &ivec4Expression)
{
    evaluate("vec4", ivec4Expression);
}

void ConstantFoldingExpressionTest::evaluateFloat(const std::string &floatExpression)
{
    evaluate("float", floatExpression);
}

void ConstantFoldingExpressionTest::evaluateInt(const std::string &intExpression)
{
    evaluate("int", intExpression);
}

void ConstantFoldingExpressionTest::evaluateUint(const std::string &uintExpression)
{
    evaluate("uint", uintExpression);
}
