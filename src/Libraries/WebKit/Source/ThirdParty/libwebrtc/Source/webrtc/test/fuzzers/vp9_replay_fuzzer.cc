/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "test/fuzzers/utils/rtp_replayer.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  auto stream_state = std::make_unique<test::RtpReplayer::StreamState>();
  VideoReceiveStreamInterface::Config vp9_config(&(stream_state->transport));

  VideoReceiveStreamInterface::Decoder vp9_decoder;
  vp9_decoder.video_format = SdpVideoFormat::VP9Profile0();
  vp9_decoder.payload_type = 124;
  vp9_config.decoders.push_back(std::move(vp9_decoder));

  vp9_config.rtp.local_ssrc = 7731;
  vp9_config.rtp.remote_ssrc = 1337;
  vp9_config.rtp.rtx_ssrc = 100;
  vp9_config.rtp.nack.rtp_history_ms = 1000;

  std::vector<VideoReceiveStreamInterface::Config> replay_configs;
  replay_configs.push_back(std::move(vp9_config));

  test::RtpReplayer::Replay(std::move(stream_state), std::move(replay_configs),
                            data, size);
}

}  // namespace webrtc
