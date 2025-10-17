/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "api/media_stream_interface.h"

#include "api/audio_options.h"
#include "api/media_types.h"
#include "api/scoped_refptr.h"

namespace webrtc {

const char* const MediaStreamTrackInterface::kVideoKind =
    cricket::kMediaTypeVideo;
const char* const MediaStreamTrackInterface::kAudioKind =
    cricket::kMediaTypeAudio;

VideoTrackInterface::ContentHint VideoTrackInterface::content_hint() const {
  return ContentHint::kNone;
}

bool AudioTrackInterface::GetSignalLevel(int* /* level */) {
  return false;
}

rtc::scoped_refptr<AudioProcessorInterface>
AudioTrackInterface::GetAudioProcessor() {
  return nullptr;
}

const cricket::AudioOptions AudioSourceInterface::options() const {
  return {};
}

}  // namespace webrtc
