/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

//===--- DefToStringsConverterTests.cpp -----------------------------------===//
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

#include "LocalizationTest.h"
#include "language/Localization/LocalizationFormat.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/ADT/StringExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/FileSystem.h"
#include "toolchain/Support/Path.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <random>
#include <string>
#include <system_error>

using namespace language;
using namespace language::diag;
using namespace language::unittests;

static std::string getMainExecutablePath() {
  toolchain::StringRef libPath = toolchain::sys::path::parent_path(LANGUAGELIB_DIR);
  toolchain::SmallString<128> MainExecutablePath(libPath);
  toolchain::sys::path::remove_filename(MainExecutablePath); // Remove /lib
  toolchain::sys::path::remove_filename(MainExecutablePath); // Remove /.
  return std::string(MainExecutablePath);
}

static std::string getDefaultLocalizationPath() {
  toolchain::SmallString<128> DefaultDiagnosticMessagesDir(getMainExecutablePath());
  toolchain::sys::path::append(DefaultDiagnosticMessagesDir, "share", "language",
                          "diagnostics");
  return std::string(DefaultDiagnosticMessagesDir);
}

TEST_F(LocalizationTest, MissingLocalizationFiles) {
  ASSERT_TRUE(toolchain::sys::fs::exists(getDefaultLocalizationPath()));
  toolchain::SmallString<128> EnglishLocalization(getDefaultLocalizationPath());
  toolchain::sys::path::append(EnglishLocalization, "en");
  toolchain::sys::path::replace_extension(EnglishLocalization, ".strings");
  ASSERT_TRUE(toolchain::sys::fs::exists(EnglishLocalization));
  toolchain::sys::path::replace_extension(EnglishLocalization, ".db");
  ASSERT_TRUE(toolchain::sys::fs::exists(EnglishLocalization));
}

TEST_F(LocalizationTest, ConverterTestMatchDiagnosticMessagesSequentially) {
  StringsLocalizationProducer strings(DiagsPath);
  strings.forEachAvailable([](language::DiagID id, toolchain::StringRef translation) {
    toolchain::StringRef msg = diagnosticMessages[static_cast<uint32_t>(id)];
    ASSERT_EQ(msg, translation);
  });
}

TEST_F(LocalizationTest, ConverterTestMatchDiagnosticMessagesRandomly) {
  StringsLocalizationProducer strings(DiagsPath);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(50, LocalDiagID::NumDiags);
  unsigned numberOfQueries = distr(gen);
  while (numberOfQueries--) {
    unsigned randomNum = RandNumber(LocalDiagID::NumDiags);
    DiagID randomId = static_cast<DiagID>(randomNum);
    toolchain::StringRef msg = diagnosticMessages[randomNum];
    toolchain::StringRef translation = strings.getMessageOr(randomId, "");
    ASSERT_EQ(msg, translation);
  }
}
