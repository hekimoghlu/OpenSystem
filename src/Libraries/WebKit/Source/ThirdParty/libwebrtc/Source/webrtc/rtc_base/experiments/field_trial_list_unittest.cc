/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "rtc_base/experiments/field_trial_list.h"

#include "absl/strings/string_view.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

using testing::ElementsAre;
using testing::IsEmpty;

namespace webrtc {

struct Garment {
  int price = 0;
  std::string color = "";
  bool has_glitter = false;

  // Only needed for testing.
  Garment() = default;
  Garment(int p, absl::string_view c, bool g)
      : price(p), color(c), has_glitter(g) {}

  bool operator==(const Garment& other) const {
    return price == other.price && color == other.color &&
           has_glitter == other.has_glitter;
  }
};

TEST(FieldTrialListTest, ParsesListParameter) {
  FieldTrialList<int> my_list("l", {5});
  EXPECT_THAT(my_list.Get(), ElementsAre(5));
  // If one element is invalid the list is unchanged.
  ParseFieldTrial({&my_list}, "l:1|2|hat");
  EXPECT_THAT(my_list.Get(), ElementsAre(5));
  ParseFieldTrial({&my_list}, "l");
  EXPECT_THAT(my_list.Get(), IsEmpty());
  ParseFieldTrial({&my_list}, "l:1|2|3");
  EXPECT_THAT(my_list.Get(), ElementsAre(1, 2, 3));
  ParseFieldTrial({&my_list}, "l:-1");
  EXPECT_THAT(my_list.Get(), ElementsAre(-1));

  FieldTrialList<std::string> another_list("l", {"hat"});
  EXPECT_THAT(another_list.Get(), ElementsAre("hat"));
  ParseFieldTrial({&another_list}, "l");
  EXPECT_THAT(another_list.Get(), IsEmpty());
  ParseFieldTrial({&another_list}, "l:");
  EXPECT_THAT(another_list.Get(), ElementsAre(""));
  ParseFieldTrial({&another_list}, "l:scarf|hat|mittens");
  EXPECT_THAT(another_list.Get(), ElementsAre("scarf", "hat", "mittens"));
  ParseFieldTrial({&another_list}, "l:scarf");
  EXPECT_THAT(another_list.Get(), ElementsAre("scarf"));
}

// Normal usage.
TEST(FieldTrialListTest, ParsesStructList) {
  FieldTrialStructList<Garment> my_list(
      {FieldTrialStructMember("color", [](Garment* g) { return &g->color; }),
       FieldTrialStructMember("price", [](Garment* g) { return &g->price; }),
       FieldTrialStructMember("has_glitter",
                              [](Garment* g) { return &g->has_glitter; })},
      {{1, "blue", false}, {2, "red", true}});

  ParseFieldTrial({&my_list},
                  "color:mauve|red|gold,"
                  "price:10|20|30,"
                  "has_glitter:1|0|1,"
                  "other_param:asdf");

  ASSERT_THAT(my_list.Get(),
              ElementsAre(Garment{10, "mauve", true}, Garment{20, "red", false},
                          Garment{30, "gold", true}));
}

// One FieldTrialList has the wrong length, so we use the user-provided default
// list.
TEST(FieldTrialListTest, StructListKeepsDefaultWithMismatchingLength) {
  FieldTrialStructList<Garment> my_list(
      {FieldTrialStructMember("wrong_length",
                              [](Garment* g) { return &g->color; }),
       FieldTrialStructMember("price", [](Garment* g) { return &g->price; })},
      {{1, "blue", true}, {2, "red", false}});

  ParseFieldTrial({&my_list},
                  "wrong_length:mauve|magenta|chartreuse|indigo,"
                  "garment:hat|hat|crown,"
                  "price:10|20|30");

  ASSERT_THAT(my_list.Get(),
              ElementsAre(Garment{1, "blue", true}, Garment{2, "red", false}));
}

// One list is missing. We set the values we're given, and the others remain
// as whatever the Garment default constructor set them to.
TEST(FieldTrialListTest, StructListUsesDefaultForMissingList) {
  FieldTrialStructList<Garment> my_list(
      {FieldTrialStructMember("color", [](Garment* g) { return &g->color; }),
       FieldTrialStructMember("price", [](Garment* g) { return &g->price; })},
      {{1, "blue", true}, {2, "red", false}});

  ParseFieldTrial({&my_list}, "price:10|20|30");

  ASSERT_THAT(my_list.Get(),
              ElementsAre(Garment{10, "", false}, Garment{20, "", false},
                          Garment{30, "", false}));
}

// The user haven't provided values for any lists, so we use the default list.
TEST(FieldTrialListTest, StructListUsesDefaultListWithoutValues) {
  FieldTrialStructList<Garment> my_list(
      {FieldTrialStructMember("color", [](Garment* g) { return &g->color; }),
       FieldTrialStructMember("price", [](Garment* g) { return &g->price; })},
      {{1, "blue", true}, {2, "red", false}});

  ParseFieldTrial({&my_list}, "");

  ASSERT_THAT(my_list.Get(),
              ElementsAre(Garment{1, "blue", true}, Garment{2, "red", false}));
}

// Some lists are provided and all are empty, so we return a empty list.
TEST(FieldTrialListTest, StructListHandlesEmptyLists) {
  FieldTrialStructList<Garment> my_list(
      {FieldTrialStructMember("color", [](Garment* g) { return &g->color; }),
       FieldTrialStructMember("price", [](Garment* g) { return &g->price; })},
      {{1, "blue", true}, {2, "red", false}});

  ParseFieldTrial({&my_list}, "color,price");

  ASSERT_EQ(my_list.Get().size(), 0u);
}

}  // namespace webrtc
