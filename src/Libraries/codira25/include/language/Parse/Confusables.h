/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

//===--- Confusables.h - Codira Confusable Character Diagnostics -*- C++ -*-===//
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

#ifndef LANGUAGE_CONFUSABLES_H
#define LANGUAGE_CONFUSABLES_H

#include "toolchain/ADT/StringRef.h"
#include <stdint.h>

namespace language {
namespace confusable {
  /// Given a UTF-8 codepoint, determines whether it appears on the Unicode
  /// specification table of confusable characters and maps to punctuation,
  /// and either returns either the expected ASCII character or 0.
  char tryConvertConfusableCharacterToASCII(uint32_t codepoint);

  /// Given a UTF-8 codepoint which is previously determined to be confusable,
  /// return the name of the confusable character and the name of the base
  /// character.
  std::pair<toolchain::StringRef, toolchain::StringRef>
  getConfusableAndBaseCodepointNames(uint32_t codepoint);
}
}

#endif
