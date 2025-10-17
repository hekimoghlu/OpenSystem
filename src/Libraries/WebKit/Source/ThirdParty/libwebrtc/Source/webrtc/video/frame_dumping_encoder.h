/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#ifndef VIDEO_FRAME_DUMPING_ENCODER_H_
#define VIDEO_FRAME_DUMPING_ENCODER_H_

#include <memory>

#include "api/field_trials_view.h"
#include "api/video_codecs/video_encoder.h"

namespace webrtc {

// Creates an encoder that wraps another passed encoder and dumps its encoded
// frames out into a unique IVF file into the directory specified by the
// "WebRTC-EncoderDataDumpDirectory" field trial. Each file generated is
// suffixed by the simulcast index of the encoded frames. If the passed encoder
// is nullptr, or the field trial is not setup, the function just returns the
// passed encoder. The directory specified by the field trial parameter should
// be delimited by ';'.
std::unique_ptr<VideoEncoder> MaybeCreateFrameDumpingEncoderWrapper(
    std::unique_ptr<VideoEncoder> encoder,
    const FieldTrialsView& field_trials);

}  // namespace webrtc

#endif  // VIDEO_FRAME_DUMPING_ENCODER_H_
