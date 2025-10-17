/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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
#ifndef VIDEO_RTP_STREAMS_SYNCHRONIZER2_H_
#define VIDEO_RTP_STREAMS_SYNCHRONIZER2_H_

#include <memory>

#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "video/stream_synchronization.h"

namespace webrtc {

class Syncable;

namespace internal {

// RtpStreamsSynchronizer is responsible for synchronizing audio and video for
// a given audio receive stream and video receive stream.
class RtpStreamsSynchronizer {
 public:
  RtpStreamsSynchronizer(TaskQueueBase* main_queue, Syncable* syncable_video);
  ~RtpStreamsSynchronizer();

  void ConfigureSync(Syncable* syncable_audio);

  // Gets the estimated playout NTP timestamp for the video frame with
  // `rtp_timestamp` and the sync offset between the current played out audio
  // frame and the video frame. Returns true on success, false otherwise.
  // The `estimated_freq_khz` is the frequency used in the RTP to NTP timestamp
  // conversion.
  bool GetStreamSyncOffsetInMs(uint32_t rtp_timestamp,
                               int64_t render_time_ms,
                               int64_t* video_playout_ntp_ms,
                               int64_t* stream_offset_ms,
                               double* estimated_freq_khz) const;

 private:
  void UpdateDelay();

  TaskQueueBase* const task_queue_;

  // Used to check if we're running on the main thread/task queue.
  // The reason we currently don't use RTC_DCHECK_RUN_ON(task_queue_) is because
  // we might be running on an rtc::Thread implementation of TaskQueue, which
  // does not consistently set itself as the active TaskQueue.
  // Instead, we rely on a SequenceChecker for now.
  RTC_NO_UNIQUE_ADDRESS SequenceChecker main_checker_;

  Syncable* const syncable_video_;

  Syncable* syncable_audio_ RTC_GUARDED_BY(main_checker_) = nullptr;
  std::unique_ptr<StreamSynchronization> sync_ RTC_GUARDED_BY(main_checker_);
  StreamSynchronization::Measurements audio_measurement_
      RTC_GUARDED_BY(main_checker_);
  StreamSynchronization::Measurements video_measurement_
      RTC_GUARDED_BY(main_checker_);
  RepeatingTaskHandle repeating_task_ RTC_GUARDED_BY(main_checker_);
  int64_t last_stats_log_ms_ RTC_GUARDED_BY(&main_checker_);
};

}  // namespace internal
}  // namespace webrtc

#endif  // VIDEO_RTP_STREAMS_SYNCHRONIZER2_H_
