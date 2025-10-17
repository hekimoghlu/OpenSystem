/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#ifndef VIDEO_END_TO_END_TESTS_MULTI_STREAM_TESTER_H_
#define VIDEO_END_TO_END_TESTS_MULTI_STREAM_TESTER_H_

#include <map>
#include <memory>

#include "api/task_queue/task_queue_base.h"
#include "call/call.h"
#include "test/direct_transport.h"
#include "test/frame_generator_capturer.h"

namespace webrtc {
// Test sets up a Call multiple senders with different resolutions and SSRCs.
// Another is set up to receive all three of these with different renderers.
class MultiStreamTester {
 public:
  static constexpr size_t kNumStreams = 3;
  const uint8_t kVideoPayloadType = 124;
  const std::map<uint8_t, MediaType> payload_type_map_ = {
      {kVideoPayloadType, MediaType::VIDEO}};

  struct CodecSettings {
    uint32_t ssrc;
    int width;
    int height;
  } codec_settings[kNumStreams];

  MultiStreamTester();

  virtual ~MultiStreamTester();

  void RunTest();

 protected:
  virtual void Wait() = 0;
  // Note: frame_generator is a point-to-pointer, since the actual instance
  // hasn't been created at the time of this call. Only when packets/frames
  // start flowing should this be dereferenced.
  virtual void UpdateSendConfig(size_t stream_index,
                                VideoSendStream::Config* send_config,
                                VideoEncoderConfig* encoder_config,
                                test::FrameGeneratorCapturer** frame_generator);
  virtual void UpdateReceiveConfig(
      size_t stream_index,
      VideoReceiveStreamInterface::Config* receive_config);
  virtual std::unique_ptr<test::DirectTransport> CreateSendTransport(
      TaskQueueBase* task_queue,
      Call* sender_call);
  virtual std::unique_ptr<test::DirectTransport> CreateReceiveTransport(
      TaskQueueBase* task_queue,
      Call* receiver_call);
};
}  // namespace webrtc
#endif  // VIDEO_END_TO_END_TESTS_MULTI_STREAM_TESTER_H_
