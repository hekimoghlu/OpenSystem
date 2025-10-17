/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#ifndef API_AUDIO_CODECS_BUILTIN_AUDIO_ENCODER_FACTORY_H_
#define API_AUDIO_CODECS_BUILTIN_AUDIO_ENCODER_FACTORY_H_

#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Creates a new factory that can create the built-in types of audio encoders.
// Note: This will link with all the code implementing those codecs, so if you
// only need a subset of the codecs, consider using
// CreateAudioEncoderFactory<...codecs listed here...>() or
// CreateOpusAudioEncoderFactory() instead.
rtc::scoped_refptr<AudioEncoderFactory> CreateBuiltinAudioEncoderFactory();

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_BUILTIN_AUDIO_ENCODER_FACTORY_H_
