/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

#ifndef COMPILER_TRANSLATOR_TREEOPS_MSL_SEPARATECOMPOUNDEXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_MSL_SEPARATECOMPOUNDEXPRESSIONS_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/msl/IdGen.h"
#include "compiler/translator/msl/SymbolEnv.h"

namespace sh
{

// Transforms code to (usually) have most one non-terminal expression per statement.
// This also rewrites (&&), (||), and (?:) into raw if/if-not/if-else statements, respectively.
//
// e.g.
//    int x = 6 + foo(y, bar());
// becomes
//    auto _1 = bar();
//    auto _2 = foo(y, _1);
//    auto _3 = 6 + _2;
//    int x = _3;
//
// WARNING:
//    - This does not rewrite object indexing operators as a whole (e.g. foo.x, foo[x]), but will
//      rewrite the arguments to the operator (when applicable).
//      e.g.
//        foo(getVec()[i + 2] + 1);
//      becomes
//        auto _1 = getVec();
//        auto _2 = i + 2;
//        auto _3 = _1[_2] + 1; // Index operator remains in (+) expr here.
//        foo(_3);
//
[[nodiscard]] bool SeparateCompoundExpressions(TCompiler &compiler,
                                               SymbolEnv &symbolEnv,
                                               IdGen &idGen,
                                               TIntermBlock &root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_MSL_SEPARATECOMPOUNDEXPRESSIONS_H_
