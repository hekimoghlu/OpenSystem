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
#include "test/pc/e2e/analyzer/video/names_collection.h"

#include <optional>
#include <string>
#include <vector>

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::Eq;
using ::testing::Ne;

TEST(NamesCollectionTest, NamesFromCtorHasUniqueIndexes) {
  NamesCollection collection(std::vector<std::string>{"alice", "bob"});

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(2)));
  EXPECT_TRUE(collection.HasName("alice"));
  EXPECT_THAT(collection.name(collection.index("alice")), Eq("alice"));

  EXPECT_TRUE(collection.HasName("bob"));
  EXPECT_THAT(collection.name(collection.index("bob")), Eq("bob"));

  EXPECT_THAT(collection.index("bob"), Ne(collection.index("alice")));
}

TEST(NamesCollectionTest, AddedNamesHasIndexes) {
  NamesCollection collection(std::vector<std::string>{});
  collection.AddIfAbsent("alice");

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_TRUE(collection.HasName("alice"));
  EXPECT_THAT(collection.name(collection.index("alice")), Eq("alice"));
}

TEST(NamesCollectionTest, AddBobDoesNotChangeAliceIndex) {
  NamesCollection collection(std::vector<std::string>{"alice"});

  size_t alice_index = collection.index("alice");

  collection.AddIfAbsent("bob");

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(2)));
  EXPECT_THAT(collection.index("alice"), Eq(alice_index));
  EXPECT_THAT(collection.index("bob"), Ne(alice_index));
}

TEST(NamesCollectionTest, AddAliceSecondTimeDoesNotChangeIndex) {
  NamesCollection collection(std::vector<std::string>{"alice"});

  size_t alice_index = collection.index("alice");

  EXPECT_THAT(collection.AddIfAbsent("alice"), Eq(alice_index));

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_THAT(collection.index("alice"), Eq(alice_index));
}

TEST(NamesCollectionTest, RemoveRemovesFromCollectionButNotIndex) {
  NamesCollection collection(std::vector<std::string>{"alice", "bob"});

  size_t bob_index = collection.index("bob");

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(2)));

  EXPECT_THAT(collection.RemoveIfPresent("bob"),
              Eq(std::optional<size_t>(bob_index)));

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_FALSE(collection.HasName("bob"));

  EXPECT_THAT(collection.index("bob"), Eq(bob_index));
  EXPECT_THAT(collection.name(bob_index), Eq("bob"));
}

TEST(NamesCollectionTest, RemoveOfAliceDoesNotChangeBobIndex) {
  NamesCollection collection(std::vector<std::string>{"alice", "bob"});

  size_t alice_index = collection.index("alice");
  size_t bob_index = collection.index("bob");

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(2)));

  EXPECT_THAT(collection.RemoveIfPresent("alice"),
              Eq(std::optional<size_t>(alice_index)));

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_THAT(collection.index("bob"), Eq(bob_index));
  EXPECT_THAT(collection.name(bob_index), Eq("bob"));
}

TEST(NamesCollectionTest, RemoveSecondTimeHasNoEffect) {
  NamesCollection collection(std::vector<std::string>{"bob"});

  size_t bob_index = collection.index("bob");

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_THAT(collection.RemoveIfPresent("bob"),
              Eq(std::optional<size_t>(bob_index)));

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(0)));
  EXPECT_THAT(collection.RemoveIfPresent("bob"), Eq(std::nullopt));
}

TEST(NamesCollectionTest, RemoveOfNotExistingHasNoEffect) {
  NamesCollection collection(std::vector<std::string>{"bob"});

  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
  EXPECT_THAT(collection.RemoveIfPresent("alice"), Eq(std::nullopt));
  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
}

TEST(NamesCollectionTest, AddRemoveAddPreserveTheIndex) {
  NamesCollection collection(std::vector<std::string>{});

  size_t alice_index = collection.AddIfAbsent("alice");
  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));

  EXPECT_THAT(collection.RemoveIfPresent("alice"),
              Eq(std::optional<size_t>(alice_index)));
  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(0)));

  EXPECT_THAT(collection.AddIfAbsent("alice"), Eq(alice_index));
  EXPECT_THAT(collection.index("alice"), Eq(alice_index));
  EXPECT_THAT(collection.size(), Eq(static_cast<size_t>(1)));
}

TEST(NamesCollectionTest, GetKnownSizeReturnsForRemovedNames) {
  NamesCollection collection(std::vector<std::string>{});

  size_t alice_index = collection.AddIfAbsent("alice");
  EXPECT_THAT(collection.GetKnownSize(), Eq(static_cast<size_t>(1)));

  EXPECT_THAT(collection.RemoveIfPresent("alice"),
              Eq(std::optional<size_t>(alice_index)));
  EXPECT_THAT(collection.GetKnownSize(), Eq(static_cast<size_t>(1)));
}

}  // namespace
}  // namespace webrtc
