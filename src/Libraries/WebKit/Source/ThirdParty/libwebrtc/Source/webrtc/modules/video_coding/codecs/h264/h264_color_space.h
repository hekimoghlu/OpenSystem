/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef MODULES_VIDEO_CODING_CODECS_H264_H264_COLOR_SPACE_H_
#define MODULES_VIDEO_CODING_CODECS_H264_H264_COLOR_SPACE_H_

// Everything declared in this header is only required when WebRTC is
// build with H264 support, please do not move anything out of the
// #ifdef unless needed and tested.
#ifdef WEBRTC_USE_H264

#if defined(WEBRTC_WIN) && !defined(__clang__)
#error "See: bugs.webrtc.org/9213#c13."
#endif

extern "C" {
#include <libavcodec/avcodec.h>
}  // extern "C"

#include "api/video/color_space.h"

namespace webrtc {

// Helper class for extracting color space information from H264 stream.
ColorSpace ExtractH264ColorSpace(AVCodecContext* codec);

}  // namespace webrtc

#endif  // WEBRTC_USE_H264

#endif  // MODULES_VIDEO_CODING_CODECS_H264_H264_COLOR_SPACE_H_
