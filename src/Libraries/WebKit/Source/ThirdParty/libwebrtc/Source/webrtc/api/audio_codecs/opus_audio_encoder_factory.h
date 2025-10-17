/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#ifndef API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_FACTORY_H_
#define API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_FACTORY_H_

#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Creates a new factory that can create only Opus audio encoders. Works like
// CreateAudioEncoderFactory<AudioEncoderOpus>(), but is easier to use and is
// not inline because it isn't a template.
rtc::scoped_refptr<AudioEncoderFactory> CreateOpusAudioEncoderFactory();

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_FACTORY_H_
