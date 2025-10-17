/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#ifndef TEST_INTEROP_CXX_STATIC_INPUTS_STATIC_VAR_H
#define TEST_INTEROP_CXX_STATIC_INPUTS_STATIC_VAR_H

static int staticVar = 2;
inline void setStaticVarFromCxx(int newVal) { staticVar = newVal; }
inline int getStaticVarFromCxx() { return staticVar; }

inline int inlineMakeStaticVar() { return 8; }
static int staticVarInlineInit = inlineMakeStaticVar();

int makeStaticVar();
static int staticVarInit = makeStaticVar();

static const int staticConst = 4;

inline int inlineMakeStaticConst() { return 16; }
static const int staticConstInlineInit = inlineMakeStaticConst();

int makeStaticConst();
static const int staticConstInit = makeStaticConst();

constexpr int makeStaticConstexpr() { return 32; }
static constexpr int staticConstexpr = makeStaticConstexpr();

class NonTrivial {
public:
  explicit NonTrivial(int val) : val(val) {}
  constexpr NonTrivial(int val, int val2) : val(val + val2) {}
  int val;
};

static NonTrivial staticNonTrivial = NonTrivial(1024);
inline void setstaticNonTrivialFromCxx(int newVal) {
  staticNonTrivial = NonTrivial(newVal);
}
inline NonTrivial *getstaticNonTrivialFromCxx() { return &staticNonTrivial; }

static const NonTrivial staticConstNonTrivial = NonTrivial(2048);
inline const NonTrivial *getstaticConstNonTrivialFromCxx() {
  return &staticConstNonTrivial;
}

static constexpr NonTrivial staticConstexprNonTrivial = NonTrivial(4096, 4096);

#endif
