/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

//===--- DeclNameLoc.cpp - Declaration Name Location Info -----------------===//
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
//  This file implements the DeclNameLoc class.
//
//===----------------------------------------------------------------------===//

#include "language/AST/DeclNameLoc.h"
#include "language/AST/ASTContext.h"
#include "language/Basic/Assertions.h"

using namespace language;

DeclNameLoc::DeclNameLoc(ASTContext &ctx, SourceLoc baseNameLoc,
                         SourceLoc lParenLoc,
                         ArrayRef<SourceLoc> argumentLabelLocs,
                         SourceLoc rParenLoc)
  : NumArgumentLabels(argumentLabelLocs.size()) {
  assert(NumArgumentLabels > 0 && "Use other constructor");

  // Copy the location information into permanent storage.
  auto storedLocs = ctx.Allocate<SourceLoc>(NumArgumentLabels + 3);
  storedLocs[BaseNameIndex] = baseNameLoc;
  storedLocs[LParenIndex] = lParenLoc;
  storedLocs[RParenIndex] = rParenLoc;
  std::memcpy(storedLocs.data() + FirstArgumentLabelIndex,
              argumentLabelLocs.data(),
              argumentLabelLocs.size() * sizeof(SourceLoc));

  LocationInfo = storedLocs.data();
}
