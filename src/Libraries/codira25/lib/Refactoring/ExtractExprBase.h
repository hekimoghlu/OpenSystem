/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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

#ifndef LANGUAGE_REFACTORING_EXTRACTEXPRBASE_H
#define LANGUAGE_REFACTORING_EXTRACTEXPRBASE_H

#include "language/AST/ASTContext.h"
#include "language/AST/SourceFile.h"
#include "language/IDE/IDERequests.h"

namespace language {
namespace refactoring {

using namespace language::ide;

enum class CannotExtractReason {
  Literal,
  VoidType,
};

class ExtractCheckResult {
  bool KnownFailure;
  SmallVector<CannotExtractReason, 2> AllReasons;

public:
  ExtractCheckResult() : KnownFailure(true) {}
  ExtractCheckResult(ArrayRef<CannotExtractReason> AllReasons)
      : KnownFailure(false), AllReasons(AllReasons.begin(), AllReasons.end()) {}
  bool success() { return success({}); }
  bool success(ArrayRef<CannotExtractReason> ExpectedReasons) {
    if (KnownFailure)
      return false;
    bool Result = true;

    // Check if any reasons aren't covered by the list of expected reasons
    // provided by the client.
    for (auto R : AllReasons) {
      Result &= toolchain::is_contained(ExpectedReasons, R);
    }
    return Result;
  }
};

/// Check whether a given range can be extracted.
/// Return true on successful condition checking,.
/// Return false on failed conditions.
ExtractCheckResult checkExtractConditions(const ResolvedRangeInfo &RangeInfo,
                                          DiagnosticEngine &DiagEngine);

class RefactoringActionExtractExprBase {
  SourceFile *TheFile;
  ResolvedRangeInfo RangeInfo;
  DiagnosticEngine &DiagEngine;
  const bool ExtractRepeated;
  StringRef PreferredName;
  SourceEditConsumer &EditConsumer;

  ASTContext &Ctx;
  SourceManager &SM;

public:
  RefactoringActionExtractExprBase(SourceFile *TheFile,
                                   ResolvedRangeInfo RangeInfo,
                                   DiagnosticEngine &DiagEngine,
                                   bool ExtractRepeated,
                                   StringRef PreferredName,
                                   SourceEditConsumer &EditConsumer)
      : TheFile(TheFile), RangeInfo(RangeInfo), DiagEngine(DiagEngine),
        ExtractRepeated(ExtractRepeated), PreferredName(PreferredName),
        EditConsumer(EditConsumer), Ctx(TheFile->getASTContext()),
        SM(Ctx.SourceMgr) {}
  bool performChange();
};

} // namespace refactoring
} // namespace language

#endif
