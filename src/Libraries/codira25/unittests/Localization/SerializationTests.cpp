/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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

//===--- LocalizationProducerTests.cpp -------------------------------------===//
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
#include "toolchain/ADT/SmallBitVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/FileSystem.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>
#include <random>

using namespace language::diag;
using namespace language::unittests;

TEST_F(LocalizationTest, TestStringsSerialization) {
  StringsLocalizationProducer strings(DiagsPath);

  auto dbFile = createTemporaryFile("en", "db");

  // First, let's serialize English translations
  {
    SerializedLocalizationWriter writer;

    strings.forEachAvailable(
        [&writer](language::DiagID id, toolchain::StringRef translation) {
          writer.insert(id, translation);
        });

    ASSERT_FALSE(writer.emit(dbFile));
  }

  // Now, let's make sure that serialized version matches "source" YAML
  auto dbContent = toolchain::MemoryBuffer::getFile(dbFile);
  ASSERT_TRUE(dbContent);

  SerializedLocalizationProducer db(std::move(dbContent.get()));
  strings.forEachAvailable(
      [&db](language::DiagID id, toolchain::StringRef translation) {
        ASSERT_EQ(translation, db.getMessageOr(id, "<no-fallback>"));
      });
}

TEST_F(LocalizationTest, TestSerializationOfEmptyFile) {
  auto dbFile = createTemporaryFile("by", "db");
  SerializedLocalizationWriter writer;
  ASSERT_FALSE(writer.emit(dbFile));

  StringsLocalizationProducer strings(DiagsPath);

  // Reading of the empty `db` file should always return default message.
  {
    auto dbContent = toolchain::MemoryBuffer::getFile(dbFile);
    ASSERT_TRUE(dbContent);

    SerializedLocalizationProducer db(std::move(dbContent.get()));
    strings.forEachAvailable([&db](language::DiagID id,
                                   toolchain::StringRef translation) {
      ASSERT_EQ("<<<default-fallback>>>",
                db.getMessageOr(id, "<<<default-fallback>>>"));
    });
  }
}

TEST_F(LocalizationTest, TestSerializationWithGaps) {
  // Initially all of the messages are included.
  toolchain::SmallBitVector includedMessages(LocalDiagID::NumDiags, true);

  // Let's punch some holes in the diagnostic content.
  for (unsigned i = 0, n = 200; i != n; ++i) {
    unsigned position = RandNumber(LocalDiagID::NumDiags);
    includedMessages.flip(position);
  }

  StringsLocalizationProducer strings(DiagsPath);
  auto dbFile = createTemporaryFile("en", "db");

  {
    SerializedLocalizationWriter writer;

    strings.forEachAvailable(
        [&](language::DiagID id, toolchain::StringRef translation) {
          if (includedMessages.test((unsigned)id))
            writer.insert(id, translation);
        });

    ASSERT_FALSE(writer.emit(dbFile));
  }


  {
    auto dbContent = toolchain::MemoryBuffer::getFile(dbFile);
    ASSERT_TRUE(dbContent);

    SerializedLocalizationProducer db(std::move(dbContent.get()));
    strings.forEachAvailable([&](language::DiagID id,
                                 toolchain::StringRef translation) {
      auto position = (unsigned)id;

      std::string expectedMessage = includedMessages.test(position)
                                        ? std::string(translation)
                                        : "<<<default-fallback>>>";

      ASSERT_EQ(expectedMessage, db.getMessageOr(id, "<<<default-fallback>>>"));
    });
  }
}
