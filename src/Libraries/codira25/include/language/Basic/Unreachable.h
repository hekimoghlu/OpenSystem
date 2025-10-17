/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

//===--- Unreachable.h - Implements language_unreachable ---*- C++ -*-===//
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
//
//  This file defines language_unreachable, which provides the
//  functionality of toolchain_unreachable without necessarily depending on
//  the LLVM support libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_UNREACHABLE_H
#define LANGUAGE_BASIC_UNREACHABLE_H

#ifdef LANGUAGE_TOOLCHAIN_SUPPORT_IS_AVAILABLE

// The implementation when LLVM is available.

#include "toolchain/Support/ErrorHandling.h"
#define language_unreachable toolchain_unreachable

#else

// The implementation when LLVM is not available.

#include <assert.h>
#include <stdlib.h>

#include "language/Runtime/Config.h"

LANGUAGE_RUNTIME_ATTRIBUTE_NORETURN LANGUAGE_ALWAYS_INLINE
inline static void language_unreachable(const char *msg) {
  (void)msg;
  LANGUAGE_RUNTIME_BUILTIN_TRAP;
}

#endif

#endif
