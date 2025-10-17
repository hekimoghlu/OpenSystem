/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#include "rtc_base/sigslot_tester.h"

#include <string>

#include "rtc_base/third_party/sigslot/sigslot.h"
#include "test/gtest.h"

namespace rtc {

TEST(SigslotTester, TestSignal1Arg) {
  sigslot::signal1<int> source1;
  int capture1;
  SigslotTester1<int, int> slot1(&source1, &capture1);
  EXPECT_EQ(0, slot1.callback_count());

  source1.emit(10);
  EXPECT_EQ(1, slot1.callback_count());
  EXPECT_EQ(10, capture1);

  source1.emit(20);
  EXPECT_EQ(2, slot1.callback_count());
  EXPECT_EQ(20, capture1);
}

TEST(SigslotTester, TestSignal2Args) {
  sigslot::signal2<int, char> source2;
  int capture1;
  char capture2;
  SigslotTester2<int, char, int, char> slot2(&source2, &capture1, &capture2);
  EXPECT_EQ(0, slot2.callback_count());

  source2.emit(10, 'x');
  EXPECT_EQ(1, slot2.callback_count());
  EXPECT_EQ(10, capture1);
  EXPECT_EQ('x', capture2);

  source2.emit(20, 'y');
  EXPECT_EQ(2, slot2.callback_count());
  EXPECT_EQ(20, capture1);
  EXPECT_EQ('y', capture2);
}

// Since it applies for 1 and 2 args, we assume it will work for up to 5 args.

TEST(SigslotTester, TestSignalWithConstReferenceArgs) {
  sigslot::signal1<const std::string&> source1;
  std::string capture1;
  SigslotTester1<const std::string&, std::string> slot1(&source1, &capture1);
  EXPECT_EQ(0, slot1.callback_count());
  source1.emit("hello");
  EXPECT_EQ(1, slot1.callback_count());
  EXPECT_EQ("hello", capture1);
}

TEST(SigslotTester, TestSignalWithPointerToConstArgs) {
  sigslot::signal1<const std::string*> source1;
  const std::string* capture1;
  SigslotTester1<const std::string*, const std::string*> slot1(&source1,
                                                               &capture1);
  EXPECT_EQ(0, slot1.callback_count());
  source1.emit(nullptr);
  EXPECT_EQ(1, slot1.callback_count());
  EXPECT_EQ(nullptr, capture1);
}

TEST(SigslotTester, TestSignalWithConstPointerArgs) {
  sigslot::signal1<std::string* const> source1;
  std::string* capture1;
  SigslotTester1<std::string* const, std::string*> slot1(&source1, &capture1);
  EXPECT_EQ(0, slot1.callback_count());
  source1.emit(nullptr);
  EXPECT_EQ(1, slot1.callback_count());
  EXPECT_EQ(nullptr, capture1);
}

}  // namespace rtc
