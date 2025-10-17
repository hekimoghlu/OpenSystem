/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#ifndef VIDEO_QUALITY_CONVERGENCE_CONTROLLER_H_
#define VIDEO_QUALITY_CONVERGENCE_CONTROLLER_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/field_trials_view.h"
#include "api/sequence_checker.h"
#include "api/video/video_codec_type.h"
#include "video/quality_convergence_monitor.h"

namespace webrtc {

class QualityConvergenceController {
 public:
  void Initialize(int number_of_layers,
                  std::optional<int> static_qp_threshold,
                  VideoCodecType codec,
                  const FieldTrialsView& trials);

  // Add the supplied `qp` value to the detection window for specified layer.
  // `is_refresh_frame` must only be `true` if the corresponding
  // video frame is a refresh frame that is used to improve the visual quality.
  // Returns `true` if the algorithm has determined that the supplied QP values
  // have converged and reached the target quality for this layer.
  bool AddSampleAndCheckTargetQuality(int layer_index,
                                      int qp,
                                      bool is_refresh_frame);

 private:
  bool initialized_ = false;
  int number_of_layers_ = 0;
  std::vector<std::unique_ptr<QualityConvergenceMonitor>> convergence_monitors_;
  SequenceChecker sequence_checker_{SequenceChecker::kDetached};
};

}  // namespace webrtc

#endif  // VIDEO_QUALITY_CONVERGENCE_CONTROLLER_H_
