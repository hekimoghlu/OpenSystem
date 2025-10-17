/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "rtc_base/experiments/field_trial_units.h"

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/experiments/field_trial_parser.h"
#include "test/gtest.h"

namespace webrtc {
namespace {
struct DummyExperiment {
  FieldTrialParameter<DataRate> target_rate =
      FieldTrialParameter<DataRate>("t", DataRate::KilobitsPerSec(100));
  FieldTrialParameter<TimeDelta> period =
      FieldTrialParameter<TimeDelta>("p", TimeDelta::Millis(100));
  FieldTrialOptional<DataSize> max_buffer =
      FieldTrialOptional<DataSize>("b", std::nullopt);

  explicit DummyExperiment(absl::string_view field_trial) {
    ParseFieldTrial({&target_rate, &max_buffer, &period}, field_trial);
  }
};
}  // namespace

TEST(FieldTrialParserUnitsTest, FallsBackToDefaults) {
  DummyExperiment exp("");
  EXPECT_EQ(exp.target_rate.Get(), DataRate::KilobitsPerSec(100));
  EXPECT_FALSE(exp.max_buffer.GetOptional().has_value());
  EXPECT_EQ(exp.period.Get(), TimeDelta::Millis(100));
}
TEST(FieldTrialParserUnitsTest, ParsesUnitParameters) {
  DummyExperiment exp("t:300kbps,b:5bytes,p:300ms");
  EXPECT_EQ(exp.target_rate.Get(), DataRate::KilobitsPerSec(300));
  EXPECT_EQ(*exp.max_buffer.GetOptional(), DataSize::Bytes(5));
  EXPECT_EQ(exp.period.Get(), TimeDelta::Millis(300));
}
TEST(FieldTrialParserUnitsTest, ParsesDefaultUnitParameters) {
  DummyExperiment exp("t:300,b:5,p:300");
  EXPECT_EQ(exp.target_rate.Get(), DataRate::KilobitsPerSec(300));
  EXPECT_EQ(*exp.max_buffer.GetOptional(), DataSize::Bytes(5));
  EXPECT_EQ(exp.period.Get(), TimeDelta::Millis(300));
}
TEST(FieldTrialParserUnitsTest, ParsesInfinityParameter) {
  DummyExperiment exp("t:inf,p:inf");
  EXPECT_EQ(exp.target_rate.Get(), DataRate::Infinity());
  EXPECT_EQ(exp.period.Get(), TimeDelta::PlusInfinity());
}
TEST(FieldTrialParserUnitsTest, ParsesOtherUnitParameters) {
  DummyExperiment exp("t:300bps,p:0.3 seconds,b:8 bytes");
  EXPECT_EQ(exp.target_rate.Get(), DataRate::BitsPerSec(300));
  EXPECT_EQ(*exp.max_buffer.GetOptional(), DataSize::Bytes(8));
  EXPECT_EQ(exp.period.Get(), TimeDelta::Millis(300));
}
TEST(FieldTrialParserUnitsTest, IgnoresOutOfRange) {
  FieldTrialConstrained<DataRate> rate("r", DataRate::KilobitsPerSec(30),
                                       DataRate::KilobitsPerSec(10),
                                       DataRate::KilobitsPerSec(100));
  FieldTrialConstrained<TimeDelta> delta("d", TimeDelta::Millis(30),
                                         TimeDelta::Millis(10),
                                         TimeDelta::Millis(100));
  FieldTrialConstrained<DataSize> size(
      "s", DataSize::Bytes(30), DataSize::Bytes(10), DataSize::Bytes(100));
  ParseFieldTrial({&rate, &delta, &size}, "r:0,d:0,s:0");
  EXPECT_EQ(rate->kbps(), 30);
  EXPECT_EQ(delta->ms(), 30);
  EXPECT_EQ(size->bytes(), 30);
  ParseFieldTrial({&rate, &delta, &size}, "r:300,d:300,s:300");
  EXPECT_EQ(rate->kbps(), 30);
  EXPECT_EQ(delta->ms(), 30);
  EXPECT_EQ(size->bytes(), 30);
  ParseFieldTrial({&rate, &delta, &size}, "r:50,d:50,s:50");
  EXPECT_EQ(rate->kbps(), 50);
  EXPECT_EQ(delta->ms(), 50);
  EXPECT_EQ(size->bytes(), 50);
}

}  // namespace webrtc
