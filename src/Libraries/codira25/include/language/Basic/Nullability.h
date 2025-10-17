/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

//===--- Nullability.h ----------------------------------------------------===//
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

#ifndef LANGUAGE_BASIC_NULLABILITY_H
#define LANGUAGE_BASIC_NULLABILITY_H

// TODO: These macro definitions are duplicated in Visibility.h. Move
// them to a single file if we find a location that both Visibility.h and
// BridgedCodiraObject.h can import.
#if __has_feature(nullability)
// Provide macros to temporarily suppress warning about the use of
// _Nullable and _Nonnull.
#define LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS                                    \
  _Pragma("clang diagnostic push")                                             \
      _Pragma("clang diagnostic ignored \"-Wnullability-extension\"")
#define LANGUAGE_END_NULLABILITY_ANNOTATIONS _Pragma("clang diagnostic pop")

#define LANGUAGE_BEGIN_ASSUME_NONNULL _Pragma("clang assume_nonnull begin")
#define LANGUAGE_END_ASSUME_NONNULL _Pragma("clang assume_nonnull end")
#else
// #define _Nullable and _Nonnull to nothing if we're not being built
// with a compiler that supports them.
#define _Nullable
#define _Nonnull
#define _Null_unspecified
#define LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS
#define LANGUAGE_END_NULLABILITY_ANNOTATIONS
#define LANGUAGE_BEGIN_ASSUME_NONNULL
#define LANGUAGE_END_ASSUME_NONNULL
#endif

#endif
