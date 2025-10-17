/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

//===-- LayoutConstraintKind.h - Layout constraints kinds -------*- C++ -*-===//
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

#ifndef LANGUAGE_LAYOUT_CONSTRAINTKIND_H
#define LANGUAGE_LAYOUT_CONSTRAINTKIND_H

/// This header is included in a bridging header. Be *very* careful with what
/// you include here! See include caveats in `ASTBridging.h`.
#include "language/Basic/LanguageBridging.h"
#include <stdint.h>

namespace language {
/// Describes a layout constraint information.
enum class ENUM_EXTENSIBILITY_ATTR(closed) LayoutConstraintKind : uint8_t {
  // It is not a known layout constraint.
  UnknownLayout LANGUAGE_NAME("unknownLayout"),
  // It is a layout constraint representing a trivial type of a known size.
  TrivialOfExactSize LANGUAGE_NAME("trivialOfExactSize"),
  // It is a layout constraint representing a trivial type of a size known to
  // be no larger than a given size.
  TrivialOfAtMostSize LANGUAGE_NAME("trivialOfAtMostSize"),
  // It is a layout constraint representing a trivial type of an unknown size.
  Trivial LANGUAGE_NAME("trivial"),
  // It is a layout constraint representing a reference counted class instance.
  Class LANGUAGE_NAME("class"),
  // It is a layout constraint representing a reference counted native class
  // instance.
  NativeClass LANGUAGE_NAME("nativeClass"),
  // It is a layout constraint representing a reference counted object.
  RefCountedObject LANGUAGE_NAME("refCountedObject"),
  // It is a layout constraint representing a native reference counted object.
  NativeRefCountedObject LANGUAGE_NAME("nativeRefCountedObject"),
  // It is a layout constraint representing a bridge object
  BridgeObject LANGUAGE_NAME("bridgeObject"),
  // It is a layout constraint representing a trivial type of a known stride.
  TrivialStride LANGUAGE_NAME("trivialStride"),
  LastLayout = TrivialStride,
};
} // namespace language

#endif // LANGUAGE_LAYOUT_CONSTRAINTKIND_H
