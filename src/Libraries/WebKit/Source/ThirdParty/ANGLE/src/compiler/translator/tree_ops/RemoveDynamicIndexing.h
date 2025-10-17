/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
// RemoveDynamicIndexing is an AST traverser to remove dynamic indexing of non-SSBO vectors and
// matrices, replacing them with calls to functions that choose which component to return or write.
// We don't need to consider dynamic indexing in SSBO since it can be directly as part of the offset
// of RWByteAddressBuffer.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEDYNAMICINDEXING_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEDYNAMICINDEXING_H_

#include "common/angleutils.h"

#include <functional>

namespace sh
{

class TCompiler;
class TIntermNode;
class TIntermBinary;
class TSymbolTable;
class PerformanceDiagnostics;

[[nodiscard]] bool RemoveDynamicIndexingOfNonSSBOVectorOrMatrix(
    TCompiler *compiler,
    TIntermNode *root,
    TSymbolTable *symbolTable,
    PerformanceDiagnostics *perfDiagnostics);

[[nodiscard]] bool RemoveDynamicIndexingOfSwizzledVector(TCompiler *compiler,
                                                         TIntermNode *root,
                                                         TSymbolTable *symbolTable,
                                                         PerformanceDiagnostics *perfDiagnostics);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEDYNAMICINDEXING_H_
