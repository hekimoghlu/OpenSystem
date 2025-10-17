/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

//===--- ClangImporterOptionsTest.cpp -------------------------------------===//
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
#include "language/Basic/LangOptions.h"
#include "toolchain/ADT/StringRef.h"
#include "gtest/gtest.h"

static std::string remap(toolchain::StringRef path) { return "remapped"; }

TEST(ClangImporterOptions, nonPathsSkipped) {
  std::vector<std::string> args = {"-unmapped", "-another=unmapped"};
  language::ClangImporterOptions options;
  options.ExtraArgs = args;

  EXPECT_EQ(options.getRemappedExtraArgs(remap), args);
}

TEST(ClangImporterOptions, optionPairs) {
  std::vector<std::string> args = {"-unmapped",    "-another=unmapped",
                                   "-I",           "some/path",
                                   "-ivfsoverlay", "another/path"};
  language::ClangImporterOptions options;
  options.ExtraArgs = args;

  std::vector<std::string> expected = {"-unmapped",    "-another=unmapped",
                                       "-I",           "remapped",
                                       "-ivfsoverlay", "remapped"};
  EXPECT_EQ(options.getRemappedExtraArgs(remap), expected);
}

TEST(ClangImporterOptions, joinedPaths) {
  std::vector<std::string> args = {"-unmapped", "-another=unmapped",
                                   "-Isome/path",
                                   "-working-directory=another/path"};
  language::ClangImporterOptions options;
  options.ExtraArgs = args;

  std::vector<std::string> expected = {"-unmapped", "-another=unmapped",
                                       "-Iremapped",
                                       "-working-directory=remapped"};
  EXPECT_EQ(options.getRemappedExtraArgs(remap), expected);
}
