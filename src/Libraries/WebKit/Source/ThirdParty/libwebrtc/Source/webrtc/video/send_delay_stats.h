/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#ifndef VIDEO_SEND_DELAY_STATS_H_
#define VIDEO_SEND_DELAY_STATS_H_

#include <stddef.h>
#include <stdint.h>

#include <map>

#include "api/units/timestamp.h"
#include "call/video_send_stream.h"
#include "modules/include/module_common_types_public.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"
#include "video/stats_counter.h"

namespace webrtc {

// Used to collect delay stats for video streams. The class gets callbacks
// from more than one threads and internally uses a mutex for data access
// synchronization.
// TODO(bugs.webrtc.org/11993): OnSendPacket and OnSentPacket will eventually
// be called consistently on the same thread. Once we're there, we should be
// able to avoid locking (at least for the fast path).
class SendDelayStats {
 public:
  explicit SendDelayStats(Clock* clock);
  ~SendDelayStats();

  // Adds the configured ssrcs for the rtp streams.
  // Stats will be calculated for these streams.
  void AddSsrcs(const VideoSendStream::Config& config);

  // Called when a packet is sent (leaving socket).
  bool OnSentPacket(int packet_id, Timestamp time);

  // Called when a packet is sent to the transport.
  void OnSendPacket(uint16_t packet_id, Timestamp capture_time, uint32_t ssrc);

 private:
  // Map holding sent packets (mapped by sequence number).
  struct SequenceNumberOlderThan {
    bool operator()(uint16_t seq1, uint16_t seq2) const {
      return IsNewerSequenceNumber(seq2, seq1);
    }
  };
  struct Packet {
    AvgCounter* send_delay;
    Timestamp capture_time;
    Timestamp send_time;
  };

  void UpdateHistograms();
  void RemoveOld(Timestamp now) RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Clock* const clock_;
  Mutex mutex_;

  std::map<uint16_t, Packet, SequenceNumberOlderThan> packets_
      RTC_GUARDED_BY(mutex_);
  size_t num_old_packets_ RTC_GUARDED_BY(mutex_);
  size_t num_skipped_packets_ RTC_GUARDED_BY(mutex_);

  // Mapped by SSRC.
  std::map<uint32_t, AvgCounter> send_delay_counters_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc
#endif  // VIDEO_SEND_DELAY_STATS_H_
