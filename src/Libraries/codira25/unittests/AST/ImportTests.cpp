/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

//===--- ImportTests.cpp - Tests for representation of imports ------------===//
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

#include "TestContext.h"
#include "language/AST/Import.h"
#include "gtest/gtest.h"

using namespace language;
using namespace language::unittest;

namespace {
/// Helper class used to create ImportPath and hold all strings for identifiers
class ImportPathContext {
  TestContext Ctx;

public:
  ImportPathContext() = default;

  /// Helper routine for building ImportPath
  /// Build()
  /// @see ImportPathBuilder
  inline ImportPath Build(StringRef Name) noexcept {
    return ImportPath::Builder(Ctx.Ctx, Name, '.').copyTo(Ctx.Ctx);
  }
};

} // namespace

TEST(ImportPath, Comparison) {
  ImportPathContext ctx;

  /// Simple soundness check:
  EXPECT_FALSE(ctx.Build("A.B.C") < ctx.Build("A.B.C"));

  /// Check order chain:
  /// A < A.A < A.A.A < A.A.B < A.B < A.B.A < AA < B < B.A
  EXPECT_LT(ctx.Build("A"), ctx.Build("A.A"));
  EXPECT_LT(ctx.Build("A.A"), ctx.Build("A.A.A"));
  EXPECT_LT(ctx.Build("A.A.A"), ctx.Build("A.A.B"));
  EXPECT_LT(ctx.Build("A.A.B"), ctx.Build("A.B"));
  EXPECT_LT(ctx.Build("A.B"), ctx.Build("A.B.A"));
  EXPECT_LT(ctx.Build("A.B.A"), ctx.Build("AA"));
  EXPECT_LT(ctx.Build("B"), ctx.Build("B.A"));

  /// Further ImportPaths are semantically incorrect, but we must
  /// check that comparing them does not cause compiler to crash.
  EXPECT_LT(ctx.Build("."), ctx.Build("A"));
  EXPECT_LT(ctx.Build("A"), ctx.Build("AA."));
  EXPECT_LT(ctx.Build("A"), ctx.Build("AA.."));
  EXPECT_LT(ctx.Build(".A"), ctx.Build("AA"));
}
