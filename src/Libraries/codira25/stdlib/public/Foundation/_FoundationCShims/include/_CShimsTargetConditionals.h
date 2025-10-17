/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

//===----------------------------------------------------------------------===//
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

#ifndef _C_SHIMS_TARGET_CONDITIONALS_H
#define _C_SHIMS_TARGET_CONDITIONALS_H

#if __has_include(<TargetConditionals.h>)
#include <TargetConditionals.h>
#endif

#if (defined(__APPLE__) && defined(__MACH__))
#define TARGET_OS_MAC 1
#else
#define TARGET_OS_MAC 0
#endif

#if defined(__linux__)
#define TARGET_OS_LINUX 1
#else
#define TARGET_OS_LINUX 0
#endif

#if defined(__unix__)
#define TARGET_OS_BSD 1
#else
#define TARGET_OS_BSD 0
#endif

#if defined(_WIN32)
#define TARGET_OS_WINDOWS 1
#else
#define TARGET_OS_WINDOWS 0
#endif

#if defined(__wasi__)
#define TARGET_OS_WASI 1
#else
#define TARGET_OS_WASI 0
#endif

#if defined(__ANDROID__)
#define TARGET_OS_ANDROID 1
#else
#define TARGET_OS_ANDROID 0
#endif

#endif // _C_SHIMS_TARGET_CONDITIONALS_H
