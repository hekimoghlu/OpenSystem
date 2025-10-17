/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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

//===--- Debug.h - Compiler debugging helpers -------------------*- C++ -*-===//
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
#ifndef LANGUAGE_BASIC_DEBUG_H
#define LANGUAGE_BASIC_DEBUG_H

#include "toolchain/Support/Compiler.h"


/// Adds attributes to the provided method signature indicating that it is a
/// debugging helper that should never be called directly from compiler code.
/// This deprecates the method so it won't be called directly and marks it as
/// used so it won't be eliminated as dead code.
#define LANGUAGE_DEBUG_HELPER(method) \
  [[deprecated("only for use in the debugger")]] method TOOLCHAIN_ATTRIBUTE_USED

/// Declares a const void instance method with the name and parameters provided.
/// Methods declared with this macro should never be called except in the
/// debugger.
#define LANGUAGE_DEBUG_DUMPER(nameAndParams) \
  LANGUAGE_DEBUG_HELPER(void nameAndParams const)

/// Declares an instance `void dump() const` method. Methods declared with this
/// macro should never be called except in the debugger.
#define LANGUAGE_DEBUG_DUMP \
  LANGUAGE_DEBUG_DUMPER(dump())

#endif


