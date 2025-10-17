/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include "language/IDE/IDEBridging.h"
#include "toolchain/Support/raw_ostream.h"
#include <climits>

ResolvedLoc::ResolvedLoc(language::CharSourceRange range,
                         std::vector<language::CharSourceRange> labelRanges,
                         std::optional<unsigned> firstTrailingLabel,
                         LabelRangeType labelType, bool isActive,
                         ResolvedLocContext context)
    : range(range), labelRanges(labelRanges),
      firstTrailingLabel(firstTrailingLabel), labelType(labelType),
      isActive(isActive), context(context) {}

ResolvedLoc::ResolvedLoc() {}

BridgedResolvedLoc::BridgedResolvedLoc(BridgedCharSourceRange range,
                                       BridgedCharSourceRangeVector labelRanges,
                                       unsigned firstTrailingLabel,
                                       LabelRangeType labelType, bool isActive,
                                       ResolvedLocContext context)
    : resolvedLoc(
          new ResolvedLoc(range.unbridged(), labelRanges.takeUnbridged(),
                          firstTrailingLabel == UINT_MAX
                              ? std::nullopt
                              : std::optional<unsigned>(firstTrailingLabel),
                          labelType, isActive, context)) {}

BridgedResolvedLocVector::BridgedResolvedLocVector()
    : vector(new std::vector<BridgedResolvedLoc>()) {}

void BridgedResolvedLocVector::push_back(BridgedResolvedLoc Loc) {
  static_cast<std::vector<ResolvedLoc> *>(vector)->push_back(
      Loc.takeUnbridged());
}

BridgedResolvedLocVector::BridgedResolvedLocVector(void *opaqueValue)
    : vector(opaqueValue) {}

void *BridgedResolvedLocVector::getOpaqueValue() const { return vector; }
