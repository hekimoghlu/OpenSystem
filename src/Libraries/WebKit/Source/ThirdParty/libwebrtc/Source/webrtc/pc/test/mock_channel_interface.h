/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef PC_TEST_MOCK_CHANNEL_INTERFACE_H_
#define PC_TEST_MOCK_CHANNEL_INTERFACE_H_

#include <string>
#include <vector>

#include "media/base/media_channel.h"
#include "pc/channel_interface.h"
#include "test/gmock.h"

namespace cricket {

// Mock class for BaseChannel.
// Use this class in unit tests to avoid dependecy on a specific
// implementation of BaseChannel.
class MockChannelInterface : public cricket::ChannelInterface {
 public:
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(VideoChannel*, AsVideoChannel, (), (override));
  MOCK_METHOD(VoiceChannel*, AsVoiceChannel, (), (override));
  MOCK_METHOD(MediaSendChannelInterface*, media_send_channel, (), (override));
  MOCK_METHOD(VoiceMediaSendChannelInterface*,
              voice_media_send_channel,
              (),
              (override));
  MOCK_METHOD(VideoMediaSendChannelInterface*,
              video_media_send_channel,
              (),
              (override));
  MOCK_METHOD(MediaReceiveChannelInterface*,
              media_receive_channel,
              (),
              (override));
  MOCK_METHOD(VoiceMediaReceiveChannelInterface*,
              voice_media_receive_channel,
              (),
              (override));
  MOCK_METHOD(VideoMediaReceiveChannelInterface*,
              video_media_receive_channel,
              (),
              (override));
  MOCK_METHOD(absl::string_view, transport_name, (), (const, override));
  MOCK_METHOD(const std::string&, mid, (), (const, override));
  MOCK_METHOD(void, Enable, (bool), (override));
  MOCK_METHOD(void,
              SetFirstPacketReceivedCallback,
              (std::function<void()>),
              (override));
  MOCK_METHOD(void,
              SetFirstPacketSentCallback,
              (std::function<void()>),
              (override));
  MOCK_METHOD(bool,
              SetLocalContent,
              (const cricket::MediaContentDescription*,
               webrtc::SdpType,
               std::string&),
              (override));
  MOCK_METHOD(bool,
              SetRemoteContent,
              (const cricket::MediaContentDescription*,
               webrtc::SdpType,
               std::string&),
              (override));
  MOCK_METHOD(bool, SetPayloadTypeDemuxingEnabled, (bool), (override));
  MOCK_METHOD(const std::vector<StreamParams>&,
              local_streams,
              (),
              (const, override));
  MOCK_METHOD(const std::vector<StreamParams>&,
              remote_streams,
              (),
              (const, override));
  MOCK_METHOD(bool,
              SetRtpTransport,
              (webrtc::RtpTransportInternal*),
              (override));
};

}  // namespace cricket

#endif  // PC_TEST_MOCK_CHANNEL_INTERFACE_H_
