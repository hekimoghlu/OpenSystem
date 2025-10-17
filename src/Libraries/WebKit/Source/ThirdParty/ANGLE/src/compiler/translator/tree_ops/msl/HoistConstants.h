/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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

#ifndef COMPILER_TRANSLATOR_TREEOPS_MSL_HOISTCONSTANTS_H_
#define COMPILER_TRANSLATOR_TREEOPS_MSL_HOISTCONSTANTS_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/msl/IdGen.h"

namespace sh
{

// Hoists function-local constants to the global scope if their Metal sizeof meets
// `minRequiredSize`.
[[nodiscard]] bool HoistConstants(TCompiler &compiler,
                                  TIntermBlock &root,
                                  IdGen &idGen,
                                  size_t minRequiredSize);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_MSL_HOISTCONSTANTS_H_
