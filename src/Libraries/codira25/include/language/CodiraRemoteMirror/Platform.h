/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

//===-- CodiraRemoteMirror/Platform.h - Remote Mirror Platform --*-- C++ -*-===//
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

#ifndef LANGUAGE_REMOTE_MIRROR_PLATFORM_H
#define LANGUAGE_REMOTE_MIRROR_PLATFORM_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(languageRemoteMirror_EXPORTS)
# if defined(__ELF__) || defined(__WASM__)
#   define LANGUAGE_REMOTE_MIRROR_LINKAGE __attribute__((__visibility__("protected")))
# elif defined(__MACH__)
#   define LANGUAGE_REMOTE_MIRROR_LINKAGE __attribute__((__visibility__("default")))
# else
#   if defined(_WINDLL)
#     define LANGUAGE_REMOTE_MIRROR_LINKAGE __declspec(dllexport)
#   else
#     define LANGUAGE_REMOTE_MIRROR_LINKAGE
#   endif
# endif
#else
# if defined(__ELF__) || defined(__MACH__) || defined(__WASM__)
#   define LANGUAGE_REMOTE_MIRROR_LINKAGE __attribute__((__visibility__("default")))
# else
#   if defined(_WINDLL)
#     define LANGUAGE_REMOTE_MIRROR_LINKAGE __declspec(dllimport)
#   else
#     define LANGUAGE_REMOTE_MIRROR_LINKAGE
#   endif
# endif
#endif

#if defined(__clang__)
#define LANGUAGE_REMOTE_MIRROR_DEPRECATED(MSG, FIX)                               \
  __attribute__((__deprecated__(MSG, FIX)))
#else
#define LANGUAGE_REMOTE_MIRROR_DEPRECATED(MSG, FIX) [[deprecated(MSG)]]
#endif

#if defined(__cplusplus)
}
#endif

#endif



