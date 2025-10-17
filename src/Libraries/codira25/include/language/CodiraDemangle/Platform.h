/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

//===-- CodiraDemangle/Platform.h - CodiraDemangle Platform Decls -*- C++ -*-===//
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

#ifndef LANGUAGE_DEMANGLE_PLATFORM_H
#define LANGUAGE_DEMANGLE_PLATFORM_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(languageDemangle_EXPORTS)
# if defined(__ELF__)
#   define LANGUAGE_DEMANGLE_LINKAGE __attribute__((__visibility__("protected")))
# elif defined(__MACH__)
#   define LANGUAGE_DEMANGLE_LINKAGE __attribute__((__visibility__("default")))
# else
#   define LANGUAGE_DEMANGLE_LINKAGE __declspec(dllexport)
# endif
#else
# if defined(__ELF__)
#   define LANGUAGE_DEMANGLE_LINKAGE __attribute__((__visibility__("default")))
# elif defined(__MACH__)
#   define LANGUAGE_DEMANGLE_LINKAGE __attribute__((__visibility__("default")))
# else
#   define LANGUAGE_DEMANGLE_LINKAGE __declspec(dllimport)
# endif
#endif

#if defined(__cplusplus)
}
#endif

#endif


