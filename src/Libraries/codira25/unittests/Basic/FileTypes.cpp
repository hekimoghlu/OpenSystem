/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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

//===--- FileTypes.cpp - for language/Basic/FileTypes.h ----------------------===//
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

#include "language/Basic/FileTypes.h"
#include "gtest/gtest.h"

namespace {
using namespace language;
using namespace language::file_types;

static const std::vector<std::pair<std::string, ID>> ExtIDs = {
#define TYPE(NAME, ID, EXTENSION, FLAGS) {EXTENSION, TY_##ID},
#include "language/Basic/FileTypes.def"
};

TEST(FileSystem, lookupTypeFromFilename) {
  for (auto &Entry: ExtIDs) {
    // no extension, skip.
    if (Entry.first.empty())
      continue;
    // raw-sil, raw-sib, lowered-sil, and raw-toolchain-ir do not have unique
    // extensions.
    if (Entry.second == TY_RawSIL || Entry.second == TY_RawSIB ||
        Entry.second == TY_LoweredSIL || Entry.second == TY_RawTOOLCHAIN_IR)
      continue;

    std::string Filename = "Myfile." + Entry.first;
    ID Type = lookupTypeFromFilename(Filename);
    ASSERT_EQ(getTypeName(Type), getTypeName(Entry.second));
  }

  ASSERT_EQ(lookupTypeFromFilename(""), TY_INVALID);
  ASSERT_EQ(lookupTypeFromFilename("."), TY_INVALID);
  ASSERT_EQ(lookupTypeFromFilename(".."), TY_INVALID);
}

} // namespace
