/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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

//===--- SupplementaryOutputPaths.h ----------------------------*- C++ -*-===*//
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

#ifndef LANGUAGE_FRONTEND_SUPPLEMENTARYOUTPUTPATHS_H
#define LANGUAGE_FRONTEND_SUPPLEMENTARYOUTPUTPATHS_H

#include "language/Basic/FileTypes.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/IR/Function.h"

#include <string>

namespace language {
struct SupplementaryOutputPaths {
#define OUTPUT(NAME, TYPE) std::string NAME;
#include "language/Basic/SupplementaryOutputPaths.def"
#undef OUTPUT

  SupplementaryOutputPaths() = default;

  /// Apply a given function for each existing (non-empty string) supplementary output
  void forEachSetOutput(toolchain::function_ref<void(const std::string&)> fn) const {
#define OUTPUT(NAME, TYPE)                                                     \
  if (!NAME.empty())                                                           \
    fn(NAME);
#include "language/Basic/SupplementaryOutputPaths.def"
#undef OUTPUT
  }

  void forEachSetOutputAndType(
      toolchain::function_ref<void(const std::string &, file_types::ID)> fn) const {
#define OUTPUT(NAME, TYPE)                                                     \
  if (!NAME.empty())                                                           \
    fn(NAME, file_types::ID::TYPE);
#include "language/Basic/SupplementaryOutputPaths.def"
#undef OUTPUT
  }

  bool empty() const {
    return
#define OUTPUT(NAME, TYPE) NAME.empty() &&
#include "language/Basic/SupplementaryOutputPaths.def"
#undef OUTPUT
        true;
  }
};
} // namespace language

#endif // LANGUAGE_FRONTEND_SUPPLEMENTARYOUTPUTPATHS_H
