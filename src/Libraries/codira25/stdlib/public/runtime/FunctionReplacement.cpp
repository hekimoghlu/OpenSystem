/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

//===--- FunctionReplacement.cpp ------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/FunctionReplacement.h"
#include "CodiraTLSContext.h"

using namespace language;
using namespace language::runtime;

char *language::language_getFunctionReplacement(char **ReplFnPtr, char *CurrFn) {
  char *ReplFn = *ReplFnPtr;
  char *RawReplFn = ReplFn;

#if LANGUAGE_PTRAUTH
  RawReplFn = ptrauth_strip(RawReplFn, ptrauth_key_function_pointer);
#endif
  if (RawReplFn == CurrFn)
    return nullptr;

  auto &ctx = CodiraTLSContext::get();
  if (ctx.CallOriginalOfReplacedFunction) {
    ctx.CallOriginalOfReplacedFunction = false;
    return nullptr;
  }
  return ReplFn;
}

char *language::language_getOrigOfReplaceable(char **OrigFnPtr) {
  char *OrigFn = *OrigFnPtr;
  CodiraTLSContext::get().CallOriginalOfReplacedFunction = true;
  return OrigFn;
}
