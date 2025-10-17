/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#ifndef API_AUDIO_CODECS_BUILTIN_AUDIO_DECODER_FACTORY_H_
#define API_AUDIO_CODECS_BUILTIN_AUDIO_DECODER_FACTORY_H_

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Creates a new factory that can create the built-in types of audio decoders.
// Note: This will link with all the code implementing those codecs, so if you
// only need a subset of the codecs, consider using
// CreateAudioDecoderFactory<...codecs listed here...>() or
// CreateOpusAudioDecoderFactory() instead.
rtc::scoped_refptr<AudioDecoderFactory> CreateBuiltinAudioDecoderFactory();

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_BUILTIN_AUDIO_DECODER_FACTORY_H_
