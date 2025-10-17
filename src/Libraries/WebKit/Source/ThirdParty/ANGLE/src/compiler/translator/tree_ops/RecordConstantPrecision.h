/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
// During parsing, all constant expressions are folded to constant union nodes. The expressions that
// have been folded may have had precision qualifiers, which should affect the precision of the
// consuming operation. If the folded constant union nodes are written to output as such they won't
// have any precision qualifiers, and their effect on the precision of the consuming operation is
// lost.
//
// RecordConstantPrecision is an AST traverser that inspects the precision qualifiers of constants
// and hoists the constants outside the containing expression as precision qualified named variables
// in case that is required for correct precision propagation.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_RECORDCONSTANTPRECISION_H_
#define COMPILER_TRANSLATOR_TREEOPS_RECORDCONSTANTPRECISION_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

[[nodiscard]] bool RecordConstantPrecision(TCompiler *compiler,
                                           TIntermNode *root,
                                           TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_RECORDCONSTANTPRECISION_H_
