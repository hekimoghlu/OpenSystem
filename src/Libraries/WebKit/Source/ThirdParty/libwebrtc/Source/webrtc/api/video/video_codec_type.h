/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef API_VIDEO_VIDEO_CODEC_TYPE_H_
#define API_VIDEO_VIDEO_CODEC_TYPE_H_

namespace webrtc {

enum VideoCodecType {
  // There are various memset(..., 0, ...) calls in the code that rely on
  // kVideoCodecGeneric being zero.
  kVideoCodecGeneric = 0,
  kVideoCodecVP8,
  kVideoCodecVP9,
  kVideoCodecAV1,
  kVideoCodecH264,
  kVideoCodecH265,
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_CODEC_TYPE_H_
