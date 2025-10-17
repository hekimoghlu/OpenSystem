/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#ifndef MODULES_VIDEO_CODING_FEC_CONTROLLER_DEFAULT_H_
#define MODULES_VIDEO_CODING_FEC_CONTROLLER_DEFAULT_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/fec_controller.h"
#include "modules/video_coding/media_opt_util.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class FecControllerDefault : public FecController {
 public:
  FecControllerDefault(const Environment& env,
                       VCMProtectionCallback* protection_callback);
  explicit FecControllerDefault(const Environment& env);

  FecControllerDefault(const FecControllerDefault&) = delete;
  FecControllerDefault& operator=(const FecControllerDefault&) = delete;

  ~FecControllerDefault() override;

  void SetProtectionCallback(
      VCMProtectionCallback* protection_callback) override;
  void SetProtectionMethod(bool enable_fec, bool enable_nack) override;
  void SetEncodingData(size_t width,
                       size_t height,
                       size_t num_temporal_layers,
                       size_t max_payload_size) override;
  uint32_t UpdateFecRates(uint32_t estimated_bitrate_bps,
                          int actual_framerate_fps,
                          uint8_t fraction_lost,
                          std::vector<bool> loss_mask_vector,
                          int64_t round_trip_time_ms) override;
  void UpdateWithEncodedData(size_t encoded_image_length,
                             VideoFrameType encoded_image_frametype) override;
  bool UseLossVectorMask() override;
  float GetProtectionOverheadRateThreshold();

 private:
  enum { kBitrateAverageWinMs = 1000 };
  const Environment env_;
  VCMProtectionCallback* protection_callback_;
  Mutex mutex_;
  std::unique_ptr<media_optimization::VCMLossProtectionLogic> loss_prot_logic_
      RTC_GUARDED_BY(mutex_);
  size_t max_payload_size_ RTC_GUARDED_BY(mutex_);

  const float overhead_threshold_;
};

}  // namespace webrtc
#endif  // MODULES_VIDEO_CODING_FEC_CONTROLLER_DEFAULT_H_
