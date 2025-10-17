/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "system_wrappers/include/field_trial.h"

#include "rtc_base/checks.h"
#include "test/gtest.h"
#include "test/testsupport/rtc_expect_death.h"

namespace webrtc {
namespace field_trial {
#if GTEST_HAS_DEATH_TEST && RTC_DCHECK_IS_ON && !defined(WEBRTC_ANDROID) && \
    !defined(WEBRTC_EXCLUDE_FIELD_TRIAL_DEFAULT)
TEST(FieldTrialValidationTest, AcceptsValidInputs) {
  InitFieldTrialsFromString("");
  InitFieldTrialsFromString("Audio/Enabled/");
  InitFieldTrialsFromString("Audio/Enabled/Video/Disabled/");
  EXPECT_TRUE(FieldTrialsStringIsValid(""));
  EXPECT_TRUE(FieldTrialsStringIsValid("Audio/Enabled/"));
  EXPECT_TRUE(FieldTrialsStringIsValid("Audio/Enabled/Video/Disabled/"));

  // Duplicate trials with the same value is fine
  InitFieldTrialsFromString("Audio/Enabled/Audio/Enabled/");
  InitFieldTrialsFromString("Audio/Enabled/B/C/Audio/Enabled/");
  EXPECT_TRUE(FieldTrialsStringIsValid("Audio/Enabled/Audio/Enabled/"));
  EXPECT_TRUE(FieldTrialsStringIsValid("Audio/Enabled/B/C/Audio/Enabled/"));
}

TEST(FieldTrialValidationDeathTest, RejectsBadInputs) {
  // Bad delimiters
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("Audio/EnabledVideo/Disabled/"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("Audio/Enabled//Video/Disabled/"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("/Audio/Enabled/Video/Disabled/"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("Audio/Enabled/Video/Disabled"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(
      InitFieldTrialsFromString("Audio/Enabled/Video/Disabled/garbage"),
      "Invalid field trials string:");
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio/EnabledVideo/Disabled/"));
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio/Enabled//Video/Disabled/"));
  EXPECT_FALSE(FieldTrialsStringIsValid("/Audio/Enabled/Video/Disabled/"));
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio/Enabled/Video/Disabled"));
  EXPECT_FALSE(
      FieldTrialsStringIsValid("Audio/Enabled/Video/Disabled/garbage"));

  // Empty trial or group
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("Audio//"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("/Enabled/"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("//"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("//Enabled"),
                   "Invalid field trials string:");
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio//"));
  EXPECT_FALSE(FieldTrialsStringIsValid("/Enabled/"));
  EXPECT_FALSE(FieldTrialsStringIsValid("//"));
  EXPECT_FALSE(FieldTrialsStringIsValid("//Enabled"));

  // Duplicate trials with different values is not fine
  RTC_EXPECT_DEATH(InitFieldTrialsFromString("Audio/Enabled/Audio/Disabled/"),
                   "Invalid field trials string:");
  RTC_EXPECT_DEATH(
      InitFieldTrialsFromString("Audio/Enabled/B/C/Audio/Disabled/"),
      "Invalid field trials string:");
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio/Enabled/Audio/Disabled/"));
  EXPECT_FALSE(FieldTrialsStringIsValid("Audio/Enabled/B/C/Audio/Disabled/"));
}

TEST(FieldTrialMergingTest, MergesValidInput) {
  EXPECT_EQ(MergeFieldTrialsStrings("Video/Enabled/", "Audio/Enabled/"),
            "Audio/Enabled/Video/Enabled/");
  EXPECT_EQ(MergeFieldTrialsStrings("Audio/Disabled/Video/Enabled/",
                                    "Audio/Enabled/"),
            "Audio/Enabled/Video/Enabled/");
  EXPECT_EQ(
      MergeFieldTrialsStrings("Audio/Enabled/Video/Enabled/", "Audio/Enabled/"),
      "Audio/Enabled/Video/Enabled/");
  EXPECT_EQ(
      MergeFieldTrialsStrings("Audio/Enabled/Audio/Enabled/", "Video/Enabled/"),
      "Audio/Enabled/Video/Enabled/");
}

TEST(FieldTrialMergingDeathTest, DchecksBadInput) {
  RTC_EXPECT_DEATH(MergeFieldTrialsStrings("Audio/Enabled/", "garbage"),
                   "Invalid field trials string:");
}

TEST(FieldTrialMergingTest, HandlesEmptyInput) {
  EXPECT_EQ(MergeFieldTrialsStrings("", "Audio/Enabled/"), "Audio/Enabled/");
  EXPECT_EQ(MergeFieldTrialsStrings("Audio/Enabled/", ""), "Audio/Enabled/");
  EXPECT_EQ(MergeFieldTrialsStrings("", ""), "");
}
#endif  // GTEST_HAS_DEATH_TEST && RTC_DCHECK_IS_ON && !defined(WEBRTC_ANDROID)
        // && !defined(WEBRTC_EXCLUDE_FIELD_TRIAL_DEFAULT)

}  // namespace field_trial
}  // namespace webrtc
