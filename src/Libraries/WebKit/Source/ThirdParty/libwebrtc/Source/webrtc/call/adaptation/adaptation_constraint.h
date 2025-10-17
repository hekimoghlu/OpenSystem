/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#ifndef CALL_ADAPTATION_ADAPTATION_CONSTRAINT_H_
#define CALL_ADAPTATION_ADAPTATION_CONSTRAINT_H_

#include <string>

#include "api/adaptation/resource.h"
#include "call/adaptation/video_source_restrictions.h"
#include "call/adaptation/video_stream_input_state.h"

namespace webrtc {

// Adaptation constraints have the ability to prevent applying a proposed
// adaptation (expressed as restrictions before/after adaptation).
class AdaptationConstraint {
 public:
  virtual ~AdaptationConstraint();

  virtual std::string Name() const = 0;

  // TODO(https://crbug.com/webrtc/11172): When we have multi-stream adaptation
  // support, this interface needs to indicate which stream the adaptation
  // applies to.
  virtual bool IsAdaptationUpAllowed(
      const VideoStreamInputState& input_state,
      const VideoSourceRestrictions& restrictions_before,
      const VideoSourceRestrictions& restrictions_after) const = 0;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_ADAPTATION_CONSTRAINT_H_
