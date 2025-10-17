/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#include "test/call_config_utils.h"

#include "call/video_receive_stream.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(CallConfigUtils, MarshalUnmarshalProcessSameObject) {
  VideoReceiveStreamInterface::Config recv_config(nullptr);

  VideoReceiveStreamInterface::Decoder decoder;
  decoder.payload_type = 10;
  decoder.video_format.name = "test";
  decoder.video_format.parameters["99"] = "b";
  recv_config.decoders.push_back(decoder);
  recv_config.render_delay_ms = 10;
  recv_config.rtp.remote_ssrc = 100;
  recv_config.rtp.local_ssrc = 101;
  recv_config.rtp.rtcp_mode = RtcpMode::kCompound;
  recv_config.rtp.lntf.enabled = false;
  recv_config.rtp.nack.rtp_history_ms = 150;
  recv_config.rtp.red_payload_type = 50;
  recv_config.rtp.rtx_ssrc = 1000;
  recv_config.rtp.rtx_associated_payload_types[10] = 10;

  VideoReceiveStreamInterface::Config unmarshaled_config =
      ParseVideoReceiveStreamJsonConfig(
          nullptr, GenerateVideoReceiveStreamJsonConfig(recv_config));

  EXPECT_EQ(recv_config.decoders[0].payload_type,
            unmarshaled_config.decoders[0].payload_type);
  EXPECT_EQ(recv_config.decoders[0].video_format.name,
            unmarshaled_config.decoders[0].video_format.name);
  EXPECT_EQ(recv_config.decoders[0].video_format.parameters,
            unmarshaled_config.decoders[0].video_format.parameters);
  EXPECT_EQ(recv_config.render_delay_ms, unmarshaled_config.render_delay_ms);
  EXPECT_EQ(recv_config.rtp.remote_ssrc, unmarshaled_config.rtp.remote_ssrc);
  EXPECT_EQ(recv_config.rtp.local_ssrc, unmarshaled_config.rtp.local_ssrc);
  EXPECT_EQ(recv_config.rtp.rtcp_mode, unmarshaled_config.rtp.rtcp_mode);
  EXPECT_EQ(recv_config.rtp.lntf.enabled, unmarshaled_config.rtp.lntf.enabled);
  EXPECT_EQ(recv_config.rtp.nack.rtp_history_ms,
            unmarshaled_config.rtp.nack.rtp_history_ms);
  EXPECT_EQ(recv_config.rtp.red_payload_type,
            unmarshaled_config.rtp.red_payload_type);
  EXPECT_EQ(recv_config.rtp.rtx_ssrc, unmarshaled_config.rtp.rtx_ssrc);
  EXPECT_EQ(recv_config.rtp.rtx_associated_payload_types,
            unmarshaled_config.rtp.rtx_associated_payload_types);
}

}  // namespace test
}  // namespace webrtc
