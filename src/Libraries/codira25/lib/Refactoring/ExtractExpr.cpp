/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

#include "ExtractExprBase.h"
#include "RefactoringActions.h"

using namespace language::refactoring;

bool RefactoringActionExtractExpr::isApplicable(const ResolvedRangeInfo &Info,
                                                DiagnosticEngine &Diag) {
  switch (Info.Kind) {
  case RangeKind::SingleExpression:
    // We disallow extract literal expression for two reasons:
    // (1) since we print the type for extracted expression, the type of a
    // literal may print as "int2048" where it is not typically users' choice;
    // (2) Extracting one literal provides little value for users.
    return checkExtractConditions(Info, Diag).success();
  case RangeKind::PartOfExpression:
  case RangeKind::SingleDecl:
  case RangeKind::MultiTypeMemberDecl:
  case RangeKind::SingleStatement:
  case RangeKind::MultiStatement:
  case RangeKind::Invalid:
    return false;
  }
  toolchain_unreachable("unhandled kind");
}

bool RefactoringActionExtractExpr::performChange() {
  return RefactoringActionExtractExprBase(TheFile, RangeInfo, DiagEngine, false,
                                          PreferredName, EditConsumer)
      .performChange();
}
