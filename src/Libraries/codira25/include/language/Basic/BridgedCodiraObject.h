/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

//===--- BridgedCodiraObject.h - C header which defines CodiraObject --------===//
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
// This is a C header, which defines the CodiraObject header. For the C++ version
// see LanguageObjectHeader.h.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_BRIDGEDLANGUAGEOBJECT_H
#define LANGUAGE_BASIC_BRIDGEDLANGUAGEOBJECT_H

#include "language/Basic/Nullability.h"

#if defined(__OpenBSD__)
#include <sys/stdint.h>
#else
#include <stdint.h>
#endif

#if !defined(__has_feature)
#define __has_feature(feature) 0
#endif

LANGUAGE_BEGIN_NULLABILITY_ANNOTATIONS

typedef const void * _Nonnull CodiraMetatype;

/// The header of a Codira object.
///
/// This must be in sync with HeapObject, which is defined in the runtime lib.
/// It must be layout compatible with the Codira object header.
struct BridgedCodiraObject {
  CodiraMetatype metatype;
  int64_t refCounts;
};

typedef struct BridgedCodiraObject * _Nonnull CodiraObject;
typedef struct BridgedCodiraObject * _Nullable OptionalCodiraObject;

LANGUAGE_END_NULLABILITY_ANNOTATIONS

#endif
