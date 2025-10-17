/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#ifndef COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITECASEDECLARATIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITECASEDECLARATIONS_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"

namespace sh
{

// EXAMPLE
//    switch (expr)
//    {
//      case 0:
//        int x = 0;
//        break;
//      case 1:
//        int y = 0;
//        {
//          int z = 0;
//        }
//        break;
//    }
// Becomes
//    {
//      int x;
//      int y;
//      switch (expr)
//      {
//        case 0:
//          x = 0;
//          break;
//        case 1:
//          y = 0;
//          {
//            int z = 0;
//          }
//          break;
//      }
//    }
[[nodiscard]] bool RewriteCaseDeclarations(TCompiler &compiler, TIntermBlock &root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_MSL_REWRITECASEDECLARATIONS_H_
