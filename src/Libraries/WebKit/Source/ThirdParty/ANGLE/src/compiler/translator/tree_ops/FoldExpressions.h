/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FoldExpressions.h: Fold expressions. This may fold expressions so that the qualifier of the
// folded node differs from the qualifier of the original expression, so it needs to be done after
// parsing and validation of qualifiers is complete. Expressions that are folded: 1. Ternary ops
// with a constant condition.

#ifndef COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TDiagnostics;

[[nodiscard]] bool FoldExpressions(TCompiler *compiler,
                                   TIntermBlock *root,
                                   TDiagnostics *diagnostics);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_
