/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#ifndef TEST_VIDEO_TEST_CONSTANTS_H_
#define TEST_VIDEO_TEST_CONSTANTS_H_

#include <cstdint>

#include "api/units/time_delta.h"

namespace webrtc {
namespace test {

class VideoTestConstants {
 public:
  static constexpr size_t kNumSsrcs = 6;
  static constexpr int kNumSimulcastStreams = 3;
  static constexpr int kDefaultWidth = 320;
  static constexpr int kDefaultHeight = 180;
  static constexpr int kDefaultFramerate = 30;
  static constexpr TimeDelta kDefaultTimeout = TimeDelta::Seconds(30);
  static constexpr TimeDelta kLongTimeout = TimeDelta::Seconds(120);
  enum classPayloadTypes : uint8_t {
    kSendRtxPayloadType = 98,
    kRtxRedPayloadType = 99,
    kVideoSendPayloadType = 100,
    kAudioSendPayloadType = 103,
    kPayloadTypeH265 = 117,
    kRedPayloadType = 118,
    kUlpfecPayloadType = 119,
    kFlexfecPayloadType = 120,
    kPayloadTypeH264 = 122,
    kPayloadTypeVP8 = 123,
    kPayloadTypeVP9 = 124,
    kPayloadTypeGeneric = 125,
    kFakeVideoSendPayloadType = 126,
  };
  static constexpr uint32_t kSendRtxSsrcs[kNumSsrcs] = {
      0xBADCAFD, 0xBADCAFE, 0xBADCAFF, 0xBADCB00, 0xBADCB01, 0xBADCB02};
  static constexpr uint32_t kVideoSendSsrcs[kNumSsrcs] = {
      0xC0FFED, 0xC0FFEE, 0xC0FFEF, 0xC0FFF0, 0xC0FFF1, 0xC0FFF2};
  static constexpr uint32_t kAudioSendSsrc = 0xDEADBEEF;
  static constexpr uint32_t kFlexfecSendSsrc = 0xBADBEEF;
  static constexpr uint32_t kReceiverLocalVideoSsrc = 0x123456;
  static constexpr uint32_t kReceiverLocalAudioSsrc = 0x1234567;
  static constexpr int kNackRtpHistoryMs = 1000;

 private:
  VideoTestConstants() = default;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_VIDEO_TEST_CONSTANTS_H_
