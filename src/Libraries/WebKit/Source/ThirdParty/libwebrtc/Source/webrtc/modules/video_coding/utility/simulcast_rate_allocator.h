/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_SIMULCAST_RATE_ALLOCATOR_H_
#define MODULES_VIDEO_CODING_UTILITY_SIMULCAST_RATE_ALLOCATOR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/environment/environment.h"
#include "api/video/video_bitrate_allocation.h"
#include "api/video/video_bitrate_allocator.h"
#include "api/video_codecs/video_codec.h"
#include "rtc_base/experiments/rate_control_settings.h"
#include "rtc_base/experiments/stable_target_rate_experiment.h"

namespace webrtc {

class SimulcastRateAllocator : public VideoBitrateAllocator {
 public:
  SimulcastRateAllocator(const Environment& env, const VideoCodec& codec);
  ~SimulcastRateAllocator() override;

  SimulcastRateAllocator(const SimulcastRateAllocator&) = delete;
  SimulcastRateAllocator& operator=(const SimulcastRateAllocator&) = delete;

  VideoBitrateAllocation Allocate(
      VideoBitrateAllocationParameters parameters) override;
  const VideoCodec& GetCodec() const;

  static float GetTemporalRateAllocation(int num_layers,
                                         int temporal_id,
                                         bool base_heavy_tl3_alloc);

  void SetLegacyConferenceMode(bool mode) override;

 private:
  void DistributeAllocationToSimulcastLayers(
      DataRate total_bitrate,
      DataRate stable_bitrate,
      VideoBitrateAllocation* allocated_bitrates);
  void DistributeAllocationToTemporalLayers(
      VideoBitrateAllocation* allocated_bitrates) const;
  std::vector<uint32_t> DefaultTemporalLayerAllocation(int bitrate_kbps,
                                                       int max_bitrate_kbps,
                                                       int simulcast_id) const;
  std::vector<uint32_t> ScreenshareTemporalLayerAllocation(
      int bitrate_kbps,
      int max_bitrate_kbps,
      int simulcast_id) const;
  int NumTemporalStreams(size_t simulcast_id) const;

  const VideoCodec codec_;
  const StableTargetRateExperiment stable_rate_settings_;
  const RateControlSettings rate_control_settings_;
  std::vector<bool> stream_enabled_;
  bool legacy_conference_mode_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_SIMULCAST_RATE_ALLOCATOR_H_
