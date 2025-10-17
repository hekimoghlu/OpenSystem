/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_TRANSPARENT_MODE_H_
#define MODULES_AUDIO_PROCESSING_AEC3_TRANSPARENT_MODE_H_

#include <memory>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

// Class for detecting and toggling the transparent mode which causes the
// suppressor to apply less suppression.
class TransparentMode {
 public:
  static std::unique_ptr<TransparentMode> Create(
      const EchoCanceller3Config& config);

  virtual ~TransparentMode() {}

  // Returns whether the transparent mode should be active.
  virtual bool Active() const = 0;

  // Resets the state of the detector.
  virtual void Reset() = 0;

  // Updates the detection decision based on new data.
  virtual void Update(int filter_delay_blocks,
                      bool any_filter_consistent,
                      bool any_filter_converged,
                      bool any_coarse_filter_converged,
                      bool all_filters_diverged,
                      bool active_render,
                      bool saturated_capture) = 0;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_PROCESSING_AEC3_TRANSPARENT_MODE_H_
