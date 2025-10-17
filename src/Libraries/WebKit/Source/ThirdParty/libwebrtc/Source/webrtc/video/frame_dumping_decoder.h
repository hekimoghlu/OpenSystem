/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef VIDEO_FRAME_DUMPING_DECODER_H_
#define VIDEO_FRAME_DUMPING_DECODER_H_

#include <memory>

#include "api/video_codecs/video_decoder.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

// Creates a decoder wrapper that writes the encoded frames to an IVF file.
std::unique_ptr<VideoDecoder> CreateFrameDumpingDecoderWrapper(
    std::unique_ptr<VideoDecoder> decoder,
    FileWrapper file);

}  // namespace webrtc

#endif  // VIDEO_FRAME_DUMPING_DECODER_H_
