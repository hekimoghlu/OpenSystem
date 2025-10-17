/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "modules/audio_coding/acm2/call_statistics.h"

#include "test/gtest.h"

namespace webrtc {

namespace acm2 {

TEST(CallStatisticsTest, InitializedZero) {
  CallStatistics call_stats;
  AudioDecodingCallStats stats;

  stats = call_stats.GetDecodingStatistics();
  EXPECT_EQ(0, stats.calls_to_neteq);
  EXPECT_EQ(0, stats.calls_to_silence_generator);
  EXPECT_EQ(0, stats.decoded_normal);
  EXPECT_EQ(0, stats.decoded_cng);
  EXPECT_EQ(0, stats.decoded_neteq_plc);
  EXPECT_EQ(0, stats.decoded_plc_cng);
  EXPECT_EQ(0, stats.decoded_muted_output);
}

TEST(CallStatisticsTest, AllCalls) {
  CallStatistics call_stats;
  AudioDecodingCallStats stats;

  call_stats.DecodedBySilenceGenerator();
  call_stats.DecodedByNetEq(AudioFrame::kNormalSpeech, false);
  call_stats.DecodedByNetEq(AudioFrame::kPLC, false);
  call_stats.DecodedByNetEq(AudioFrame::kCodecPLC, false);
  call_stats.DecodedByNetEq(AudioFrame::kPLCCNG, true);  // Let this be muted.
  call_stats.DecodedByNetEq(AudioFrame::kCNG, false);

  stats = call_stats.GetDecodingStatistics();
  EXPECT_EQ(5, stats.calls_to_neteq);
  EXPECT_EQ(1, stats.calls_to_silence_generator);
  EXPECT_EQ(1, stats.decoded_normal);
  EXPECT_EQ(1, stats.decoded_cng);
  EXPECT_EQ(1, stats.decoded_neteq_plc);
  EXPECT_EQ(1, stats.decoded_codec_plc);
  EXPECT_EQ(1, stats.decoded_plc_cng);
  EXPECT_EQ(1, stats.decoded_muted_output);
}

}  // namespace acm2

}  // namespace webrtc
