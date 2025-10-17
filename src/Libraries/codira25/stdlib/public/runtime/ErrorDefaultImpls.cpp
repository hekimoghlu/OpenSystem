/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

//===--- ErrorDefaultImpls.cpp - Error default implementations ------------===//
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
// This implements helpers for the default implementations of Error protocol
// members.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Config.h"
#include "language/Runtime/Metadata.h"
using namespace language;

// @_silgen_name("_language_stdlib_getDefaultErrorCode")
// fn _getDefaultErrorCode<T : Error>(_ x: T) -> Integer
LANGUAGE_CC(language) LANGUAGE_RUNTIME_STDLIB_SPI
intptr_t _language_stdlib_getDefaultErrorCode(OpaqueValue *error,
                                           const Metadata *T,
                                           const WitnessTable *Error) {
  intptr_t result;

  switch (T->getKind()) {
    case MetadataKind::Enum: 
      result = T->vw_getEnumTag(error);
      break;

    default:
      result = 1;
      break;
  }

  return result;
}
