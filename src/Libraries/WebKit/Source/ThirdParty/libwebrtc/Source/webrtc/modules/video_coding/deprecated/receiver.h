/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#ifndef MODULES_VIDEO_CODING_DEPRECATED_RECEIVER_H_
#define MODULES_VIDEO_CODING_DEPRECATED_RECEIVER_H_

#include <memory>
#include <vector>

#include "api/field_trials_view.h"
#include "modules/video_coding/deprecated/event_wrapper.h"
#include "modules/video_coding/deprecated/jitter_buffer.h"
#include "modules/video_coding/deprecated/packet.h"
#include "modules/video_coding/include/video_coding_defines.h"
#include "modules/video_coding/timing/timing.h"

namespace webrtc {

class Clock;
class VCMEncodedFrame;

class VCMReceiver {
 public:
  VCMReceiver(VCMTiming* timing,
              Clock* clock,
              const FieldTrialsView& field_trials);

  // Using this constructor, you can specify a different event implemetation for
  // the jitter buffer. Useful for unit tests when you want to simulate incoming
  // packets, in which case the jitter buffer's wait event is different from
  // that of VCMReceiver itself.
  VCMReceiver(VCMTiming* timing,
              Clock* clock,
              std::unique_ptr<EventWrapper> receiver_event,
              std::unique_ptr<EventWrapper> jitter_buffer_event,
              const FieldTrialsView& field_trials);

  ~VCMReceiver();

  int32_t InsertPacket(const VCMPacket& packet);
  VCMEncodedFrame* FrameForDecoding(uint16_t max_wait_time_ms,
                                    bool prefer_late_decoding);
  void ReleaseFrame(VCMEncodedFrame* frame);

  // NACK.
  void SetNackSettings(size_t max_nack_list_size,
                       int max_packet_age_to_nack,
                       int max_incomplete_time_ms);
  std::vector<uint16_t> NackList(bool* request_key_frame);

 private:
  Clock* const clock_;
  VCMJitterBuffer jitter_buffer_;
  VCMTiming* timing_;
  std::unique_ptr<EventWrapper> render_wait_event_;
  int max_video_delay_ms_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_DEPRECATED_RECEIVER_H_
