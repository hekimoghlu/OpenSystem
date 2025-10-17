/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SimplifyLoopConditions_test.cpp:
//   Tests that loop conditions are simplified.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

namespace
{

class SimplifyLoopConditionsTest : public MatchOutputCodeTest
{
  public:
    SimplifyLoopConditionsTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_ESSL_OUTPUT)
    {
        ShCompileOptions defaultCompileOptions       = {};
        defaultCompileOptions.simplifyLoopConditions = true;
        defaultCompileOptions.validateAST            = true;
        setDefaultCompileOptions(defaultCompileOptions);
    }
};

TEST_F(SimplifyLoopConditionsTest, For)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    for (;;) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  {
    bool sbba = true;
    while (sbba)
    {
      {
      }
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForExprConstant)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    for (;true;) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  {
    bool sbba = true;
    while (sbba)
    {
      {
      }
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForExprSymbol)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    bool b = true;
    for (;b;) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  bool _ub = true;
  {
    while (_ub)
    {
      {
      }
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForExpr)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    bool b = true;
    for (;b == true;) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  bool _ub = true;
  {
    bool sbbb = (_ub == true);
    while (sbbb)
    {
      {
      }
      (sbbb = (_ub == true));
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForInitExprSymbol)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    for (bool b = true; b;) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  {
    bool _ub = true;
    while (_ub)
    {
      {
      }
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForInitExprSymbolExpr2)
{
    const char kShader[]   = R"(#version 300 es
void main() {
    for (bool b = true; b; b = false) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  {
    bool _ub = true;
    while (_ub)
    {
      {
      }
      (_ub = false);
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForInitExprExpr2)
{
    const char kShader[]   = R"(#version 300 es
void main() {
        for (highp int i; i < 100; ++i) { }
})";
    const char kExpected[] = R"(#version 300 es
void main(){
  {
    highp int _ui;
    bool sbbb = (_ui < 100);
    while (sbbb)
    {
      {
      }
      (++_ui);
      (sbbb = (_ui < 100));
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForInitExprExpr2Break)
{
    const char kShader[]   = R"(#version 300 es
uniform highp int u;
void main() {
    for (highp int i; i < 100; ++i) { if (i < u) break; }
})";
    const char kExpected[] = R"(#version 300 es
uniform highp int _uu;
void main(){
  {
    highp int _ui;
    bool sbbc = (_ui < 100);
    while (sbbc)
    {
      {
        if ((_ui < _uu))
        {
          break;
        }
      }
      (++_ui);
      (sbbc = (_ui < 100));
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

TEST_F(SimplifyLoopConditionsTest, ForInitExprExpr2Continue)
{
    const char kShader[]   = R"(#version 300 es
uniform highp int u;
void main() {
    for (highp int i; i < 100; ++i) { if (i < u) continue; ++i; }
})";
    const char kExpected[] = R"(#version 300 es
uniform highp int _uu;
void main(){
  {
    highp int _ui;
    bool sbbc = (_ui < 100);
    while (sbbc)
    {
      {
        if ((_ui < _uu))
        {
          (++_ui);
          (sbbc = (_ui < 100));
          continue;
        }
        (++_ui);
      }
      (++_ui);
      (sbbc = (_ui < 100));
    }
  }
}
)";
    compile(kShader);
    EXPECT_EQ(kExpected, outputCode(SH_ESSL_OUTPUT));
}

}  // namespace