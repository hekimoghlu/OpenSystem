/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#ifndef API_AUDIO_ECHO_DETECTOR_CREATOR_H_
#define API_AUDIO_ECHO_DETECTOR_CREATOR_H_

#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Returns an instance of the WebRTC implementation of a residual echo detector.
// It can be provided to the webrtc::BuiltinAudioProcessingBuilder to obtain the
// usual residual echo metrics.
scoped_refptr<EchoDetector> CreateEchoDetector();

}  // namespace webrtc

#endif  // API_AUDIO_ECHO_DETECTOR_CREATOR_H_
