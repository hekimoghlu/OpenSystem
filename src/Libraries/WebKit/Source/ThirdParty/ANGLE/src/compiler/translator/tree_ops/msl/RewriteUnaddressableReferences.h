/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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

#ifndef COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITEUNADDRESSABLEREFERENCES_H_
#define COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITEUNADDRESSABLEREFERENCES_H_

#include "compiler/translator/Compiler.h"
#include "compiler/translator/msl/ProgramPrelude.h"
#include "compiler/translator/msl/SymbolEnv.h"

namespace sh
{

// Given:
//   void foo(out x);
// It is possible for the following to be legal in GLSL but not in Metal:
//   foo(expression);
// This can happen in cases where `expression` is a vector swizzle or vector element access.
// This rewrite functionality introduces temporaries that serve as proxies to be passed to the
// out/inout parameters as needed. The corresponding expressions get populated with their
// proxies after the function call.
[[nodiscard]] bool RewriteUnaddressableReferences(TCompiler &compiler,
                                                  TIntermBlock &root,
                                                  SymbolEnv &symbolEnv);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITEUNADDRESSABLEREFERENCES_H_
