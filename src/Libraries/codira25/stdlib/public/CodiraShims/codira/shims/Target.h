/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

//===--- Target.h - Info about the current compilation target ---*- C++ -*-===//
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
// Info about the current compilation target.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_STDLIB_SHIMS_ABI_TARGET_H
#define LANGUAGE_STDLIB_SHIMS_ABI_TARGET_H

#if !defined(__has_builtin)
#define __has_builtin(x) 0
#endif

// Is the target platform a simulator? We can't use TargetConditionals
// when included from CodiraShims, so use the builtin.
#if __has_builtin(__is_target_environment)
# if __is_target_environment(simulator)
#  define LANGUAGE_TARGET_OS_SIMULATOR 1
# else
#  define LANGUAGE_TARGET_OS_SIMULATOR 0
# endif
#endif

// Is the target platform Darwin?
#if __has_builtin(__is_target_os)
# if __is_target_os(darwin)
#   define LANGUAGE_TARGET_OS_DARWIN 1
# else
#   define LANGUAGE_TARGET_OS_DARWIN 0
# endif
#else
# define LANGUAGE_TARGET_OS_DARWIN 0
#endif

#endif // LANGUAGE_STDLIB_SHIMS_ABI_TARGET_H
