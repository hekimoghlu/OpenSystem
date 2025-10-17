/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "rtc_base/experiments/keyframe_interval_settings.h"

#include "test/explicit_key_value_config.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using test::ExplicitKeyValueConfig;

TEST(KeyframeIntervalSettingsTest, ParsesMinKeyframeSendIntervalMs) {
  EXPECT_FALSE(KeyframeIntervalSettings(ExplicitKeyValueConfig(""))
                   .MinKeyframeSendIntervalMs());

  ExplicitKeyValueConfig field_trials(
      "WebRTC-KeyframeInterval/min_keyframe_send_interval_ms:100/");
  EXPECT_EQ(KeyframeIntervalSettings(field_trials).MinKeyframeSendIntervalMs(),
            100);
}

TEST(KeyframeIntervalSettingsTest, DoesNotParseIncorrectValues) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-KeyframeInterval/min_keyframe_send_interval_ms:a/");
  EXPECT_FALSE(
      KeyframeIntervalSettings(field_trials).MinKeyframeSendIntervalMs());
}

}  // namespace
}  // namespace webrtc
