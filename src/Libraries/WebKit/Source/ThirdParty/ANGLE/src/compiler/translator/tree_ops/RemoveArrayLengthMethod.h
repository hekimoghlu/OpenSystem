/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveArrayLengthMethod.h:
//   Fold array length expressions, including cases where the "this" node has side effects.
//   Example:
//     int i = (a = b).length();
//     int j = (func()).length();
//   becomes:
//     (a = b);
//     int i = <constant array length>;
//     func();
//     int j = <constant array length>;
//
//   Must be run after SplitSequenceOperator, SimplifyLoopConditions and SeparateDeclarations steps
//   have been done to expressions containing calls of the array length method.
//
//   Does nothing to length method calls done on runtime-sized arrays.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEARRAYLENGTHMETHOD_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEARRAYLENGTHMETHOD_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;

[[nodiscard]] bool RemoveArrayLengthMethod(TCompiler *compiler, TIntermBlock *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEARRAYLENGTHMETHOD_H_
