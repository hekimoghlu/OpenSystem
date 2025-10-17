/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#ifndef VIDEO_ENCODER_RTCP_FEEDBACK_H_
#define VIDEO_ENCODER_RTCP_FEEDBACK_H_

#include <functional>
#include <vector>

#include "api/environment/environment.h"
#include "api/sequence_checker.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "call/rtp_video_sender_interface.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/system/no_unique_address.h"
#include "video/video_stream_encoder_interface.h"

namespace webrtc {

class VideoStreamEncoderInterface;

// This class passes feedback (such as key frame requests or loss notifications)
// from the RtpRtcp module.
class EncoderRtcpFeedback : public RtcpIntraFrameObserver,
                            public RtcpLossNotificationObserver {
 public:
  EncoderRtcpFeedback(
      const Environment& env,
      bool per_layer_keyframes,
      const std::vector<uint32_t>& ssrcs,
      VideoStreamEncoderInterface* encoder,
      std::function<std::vector<RtpSequenceNumberMap::Info>(
          uint32_t ssrc,
          const std::vector<uint16_t>& seq_nums)> get_packet_infos);
  ~EncoderRtcpFeedback() override = default;

  void OnReceivedIntraFrameRequest(uint32_t ssrc) override;

  // Implements RtcpLossNotificationObserver.
  void OnReceivedLossNotification(uint32_t ssrc,
                                  uint16_t seq_num_of_last_decodable,
                                  uint16_t seq_num_of_last_received,
                                  bool decodability_flag) override;

 private:
  const Environment env_;
  const std::vector<uint32_t> ssrcs_;
  const bool per_layer_keyframes_;
  const std::function<std::vector<RtpSequenceNumberMap::Info>(
      uint32_t ssrc,
      const std::vector<uint16_t>& seq_nums)>
      get_packet_infos_;
  VideoStreamEncoderInterface* const video_stream_encoder_;

  RTC_NO_UNIQUE_ADDRESS SequenceChecker packet_delivery_queue_;
  std::vector<Timestamp> time_last_packet_delivery_queue_
      RTC_GUARDED_BY(packet_delivery_queue_);

  const TimeDelta min_keyframe_send_interval_;
};

}  // namespace webrtc

#endif  // VIDEO_ENCODER_RTCP_FEEDBACK_H_
