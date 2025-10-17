/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include "api/audio/builtin_audio_processing_builder.h"

#include <utility>

#include "absl/base/nullability.h"
#include "api/audio/audio_processing.h"
#include "api/environment/environment.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "modules/audio_processing/audio_processing_impl.h"
#include "rtc_base/logging.h"

namespace webrtc {

absl::Nullable<scoped_refptr<AudioProcessing>>
BuiltinAudioProcessingBuilder::Build(const Environment& /*env*/) {
  // TODO: bugs.webrtc.org/369904700 - Pass `env` when AudioProcessingImpl gets
  // constructor that accepts it.
  return make_ref_counted<AudioProcessingImpl>(
      config_, std::move(capture_post_processing_),
      std::move(render_pre_processing_), std::move(echo_control_factory_),
      std::move(echo_detector_), std::move(capture_analyzer_));
}

}  // namespace webrtc
