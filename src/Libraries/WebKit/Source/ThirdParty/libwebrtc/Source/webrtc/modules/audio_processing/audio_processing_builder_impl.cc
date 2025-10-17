/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include <memory>

#include "api/audio/audio_processing.h"
#include "api/make_ref_counted.h"
#include "modules/audio_processing/audio_processing_impl.h"

namespace webrtc {

AudioProcessingBuilder::AudioProcessingBuilder() = default;
AudioProcessingBuilder::~AudioProcessingBuilder() = default;

rtc::scoped_refptr<AudioProcessing> AudioProcessingBuilder::Create() {
#ifdef WEBRTC_EXCLUDE_AUDIO_PROCESSING_MODULE
  // Return a null pointer when the APM is excluded from the build.
  return nullptr;
#else  // WEBRTC_EXCLUDE_AUDIO_PROCESSING_MODULE
  return rtc::make_ref_counted<AudioProcessingImpl>(
      config_, std::move(capture_post_processing_),
      std::move(render_pre_processing_), std::move(echo_control_factory_),
      std::move(echo_detector_), std::move(capture_analyzer_));
#endif
}

}  // namespace webrtc
