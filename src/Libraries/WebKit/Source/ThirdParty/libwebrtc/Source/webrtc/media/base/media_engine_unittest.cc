/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#include "media/base/media_engine.h"

#include "test/gmock.h"
#include "test/gtest.h"

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Return;
using ::testing::StrEq;
using ::webrtc::RtpExtension;
using ::webrtc::RtpHeaderExtensionCapability;
using ::webrtc::RtpTransceiverDirection;

namespace cricket {
namespace {

class MockRtpHeaderExtensionQueryInterface
    : public RtpHeaderExtensionQueryInterface {
 public:
  MOCK_METHOD(std::vector<RtpHeaderExtensionCapability>,
              GetRtpHeaderExtensions,
              (),
              (const, override));
};

}  // namespace

TEST(MediaEngineTest, ReturnsNotStoppedHeaderExtensions) {
  MockRtpHeaderExtensionQueryInterface mock;
  std::vector<RtpHeaderExtensionCapability> extensions(
      {RtpHeaderExtensionCapability("uri1", 1,
                                    RtpTransceiverDirection::kInactive),
       RtpHeaderExtensionCapability("uri2", 2,
                                    RtpTransceiverDirection::kSendRecv),
       RtpHeaderExtensionCapability("uri3", 3,
                                    RtpTransceiverDirection::kStopped),
       RtpHeaderExtensionCapability("uri4", 4,
                                    RtpTransceiverDirection::kSendOnly),
       RtpHeaderExtensionCapability("uri5", 5,
                                    RtpTransceiverDirection::kRecvOnly)});
  EXPECT_CALL(mock, GetRtpHeaderExtensions).WillOnce(Return(extensions));
  EXPECT_THAT(GetDefaultEnabledRtpHeaderExtensions(mock),
              ElementsAre(Field(&RtpExtension::uri, StrEq("uri1")),
                          Field(&RtpExtension::uri, StrEq("uri2")),
                          Field(&RtpExtension::uri, StrEq("uri4")),
                          Field(&RtpExtension::uri, StrEq("uri5"))));
}

// This class mocks methods declared as pure virtual in the interface.
// Since the tests are aiming to check the patterns of overrides, the
// functions with default implementations are not mocked.
class MostlyMockVoiceEngineInterface : public VoiceEngineInterface {
 public:
  MOCK_METHOD(std::vector<webrtc::RtpHeaderExtensionCapability>,
              GetRtpHeaderExtensions,
              (),
              (const, override));
  MOCK_METHOD(void, Init, (), (override));
  MOCK_METHOD(rtc::scoped_refptr<webrtc::AudioState>,
              GetAudioState,
              (),
              (const, override));
  MOCK_METHOD(std::vector<Codec>&, send_codecs, (), (const, override));
  MOCK_METHOD(std::vector<Codec>&, recv_codecs, (), (const, override));
  MOCK_METHOD(bool,
              StartAecDump,
              (webrtc::FileWrapper file, int64_t max_size_bytes),
              (override));
  MOCK_METHOD(void, StopAecDump, (), (override));
  MOCK_METHOD(std::optional<webrtc::AudioDeviceModule::Stats>,
              GetAudioDeviceStats,
              (),
              (override));
};

}  // namespace cricket
