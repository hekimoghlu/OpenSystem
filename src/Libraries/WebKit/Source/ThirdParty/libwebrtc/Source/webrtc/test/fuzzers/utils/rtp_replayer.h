/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#ifndef TEST_FUZZERS_UTILS_RTP_REPLAYER_H_
#define TEST_FUZZERS_UTILS_RTP_REPLAYER_H_

#include <stdio.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "api/test/video/function_video_decoder_factory.h"
#include "api/video_codecs/video_decoder.h"
#include "call/call.h"
#include "media/engine/internal_decoder_factory.h"
#include "rtc_base/fake_clock.h"
#include "rtc_base/time_utils.h"
#include "test/null_transport.h"
#include "test/rtp_file_reader.h"
#include "test/test_video_capturer.h"
#include "test/video_renderer.h"

namespace webrtc {
namespace test {

// The RtpReplayer is a utility for fuzzing the RTP/RTCP receiver stack in
// WebRTC. It achieves this by accepting a set of Receiver configurations and
// an RtpDump (consisting of both RTP and RTCP packets). The `rtp_dump` is
// passed in as a buffer to allow simple mutation fuzzing directly on the dump.
class RtpReplayer final {
 public:
  // Holds all the important stream information required to emulate the WebRTC
  // rtp receival code path.
  struct StreamState {
    test::NullTransport transport;
    std::vector<std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>>> sinks;
    std::vector<VideoReceiveStreamInterface*> receive_streams;
    std::unique_ptr<VideoDecoderFactory> decoder_factory;
  };

  // Construct an RtpReplayer from a JSON replay configuration file.
  static void Replay(const std::string& replay_config_filepath,
                     const uint8_t* rtp_dump_data,
                     size_t rtp_dump_size);

  // Construct an RtpReplayer from  a set of
  // VideoReceiveStreamInterface::Configs. Note the stream_state.transport must
  // be set for each receiver stream.
  static void Replay(
      std::unique_ptr<StreamState> stream_state,
      std::vector<VideoReceiveStreamInterface::Config> receive_stream_config,
      const uint8_t* rtp_dump_data,
      size_t rtp_dump_size);

 private:
  // Reads the replay configuration from Json.
  static std::vector<VideoReceiveStreamInterface::Config> ReadConfigFromFile(
      const std::string& replay_config,
      Transport* transport);

  // Configures the stream state based on the receiver configurations.
  static void SetupVideoStreams(
      std::vector<VideoReceiveStreamInterface::Config>* receive_stream_configs,
      StreamState* stream_state,
      Call* call);

  // Creates a new RtpReader which can read the RtpDump
  static std::unique_ptr<test::RtpFileReader> CreateRtpReader(
      const uint8_t* rtp_dump_data,
      size_t rtp_dump_size);

  // Replays each packet to from the RtpDump.
  static void ReplayPackets(rtc::FakeClock* clock,
                            Call* call,
                            test::RtpFileReader* rtp_reader,
                            const RtpHeaderExtensionMap& extensions);
};  // class RtpReplayer

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FUZZERS_UTILS_RTP_REPLAYER_H_
