/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef COMMON_VIDEO_INCLUDE_BITRATE_ADJUSTER_H_
#define COMMON_VIDEO_INCLUDE_BITRATE_ADJUSTER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "rtc_base/rate_statistics.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// Certain hardware encoders tend to consistently overshoot the bitrate that
// they are configured to encode at. This class estimates an adjusted bitrate
// that when set on the encoder will produce the desired bitrate.
class RTC_EXPORT BitrateAdjuster {
 public:
  // min_adjusted_bitrate_pct and max_adjusted_bitrate_pct are the lower and
  // upper bound outputted adjusted bitrates as a percentage of the target
  // bitrate.
  BitrateAdjuster(float min_adjusted_bitrate_pct,
                  float max_adjusted_bitrate_pct);
  virtual ~BitrateAdjuster() {}

  static const uint32_t kBitrateUpdateIntervalMs;
  static const uint32_t kBitrateUpdateFrameInterval;
  static const float kBitrateTolerancePct;
  static const float kBytesPerMsToBitsPerSecond;

  // Sets the desired bitrate in bps (bits per second).
  // Should be called at least once before Update.
  void SetTargetBitrateBps(uint32_t bitrate_bps);
  uint32_t GetTargetBitrateBps() const;

  // Returns the adjusted bitrate in bps.
  uint32_t GetAdjustedBitrateBps() const;

  // Returns what we think the current bitrate is.
  std::optional<uint32_t> GetEstimatedBitrateBps();

  // This should be called after each frame is encoded. The timestamp at which
  // it is called is used to estimate the output bitrate of the encoder.
  // Should be called from only one thread.
  void Update(size_t frame_size);

 private:
  // Returns true if the bitrate is within kBitrateTolerancePct of bitrate_bps.
  bool IsWithinTolerance(uint32_t bitrate_bps, uint32_t target_bitrate_bps);

  // Returns smallest possible adjusted value.
  uint32_t GetMinAdjustedBitrateBps() const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  // Returns largest possible adjusted value.
  uint32_t GetMaxAdjustedBitrateBps() const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void Reset();
  void UpdateBitrate(uint32_t current_time_ms)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable Mutex mutex_;
  const float min_adjusted_bitrate_pct_;
  const float max_adjusted_bitrate_pct_;
  // The bitrate we want.
  volatile uint32_t target_bitrate_bps_ RTC_GUARDED_BY(mutex_);
  // The bitrate we use to get what we want.
  volatile uint32_t adjusted_bitrate_bps_ RTC_GUARDED_BY(mutex_);
  // The target bitrate that the adjusted bitrate was computed from.
  volatile uint32_t last_adjusted_target_bitrate_bps_ RTC_GUARDED_BY(mutex_);
  // Used to estimate bitrate.
  RateStatistics bitrate_tracker_ RTC_GUARDED_BY(mutex_);
  // The last time we tried to adjust the bitrate.
  uint32_t last_bitrate_update_time_ms_ RTC_GUARDED_BY(mutex_);
  // The number of frames since the last time we tried to adjust the bitrate.
  uint32_t frames_since_last_update_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_INCLUDE_BITRATE_ADJUSTER_H_
