/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#ifndef TEST_LAYER_FILTERING_TRANSPORT_H_
#define TEST_LAYER_FILTERING_TRANSPORT_H_

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>

#include "api/call/transport.h"
#include "api/media_types.h"
#include "call/call.h"
#include "call/simulated_packet_receiver.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"
#include "test/direct_transport.h"

namespace webrtc {

namespace test {

class LayerFilteringTransport : public test::DirectTransport {
 public:
  LayerFilteringTransport(
      TaskQueueBase* task_queue,
      std::unique_ptr<SimulatedPacketReceiverInterface> pipe,
      Call* send_call,
      uint8_t vp8_video_payload_type,
      uint8_t vp9_video_payload_type,
      int selected_tl,
      int selected_sl,
      const std::map<uint8_t, MediaType>& payload_type_map,
      uint32_t ssrc_to_filter_min,
      uint32_t ssrc_to_filter_max,
      rtc::ArrayView<const RtpExtension> audio_extensions,
      rtc::ArrayView<const RtpExtension> video_extensions);
  LayerFilteringTransport(
      TaskQueueBase* task_queue,
      std::unique_ptr<SimulatedPacketReceiverInterface> pipe,
      Call* send_call,
      uint8_t vp8_video_payload_type,
      uint8_t vp9_video_payload_type,
      int selected_tl,
      int selected_sl,
      const std::map<uint8_t, MediaType>& payload_type_map,
      rtc::ArrayView<const RtpExtension> audio_extensions,
      rtc::ArrayView<const RtpExtension> video_extensions);
  bool DiscardedLastPacket() const;
  bool SendRtp(rtc::ArrayView<const uint8_t> data,
               const PacketOptions& options) override;

 private:
  // Used to distinguish between VP8 and VP9.
  const uint8_t vp8_video_payload_type_;
  const uint8_t vp9_video_payload_type_;
  const std::unique_ptr<VideoRtpDepacketizer> vp8_depacketizer_;
  const std::unique_ptr<VideoRtpDepacketizer> vp9_depacketizer_;
  // Discard or invalidate all temporal/spatial layers with id greater than the
  // selected one. -1 to disable filtering.
  const int selected_tl_;
  const int selected_sl_;
  bool discarded_last_packet_;
  int num_active_spatial_layers_;
  const uint32_t ssrc_to_filter_min_;
  const uint32_t ssrc_to_filter_max_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_LAYER_FILTERING_TRANSPORT_H_
