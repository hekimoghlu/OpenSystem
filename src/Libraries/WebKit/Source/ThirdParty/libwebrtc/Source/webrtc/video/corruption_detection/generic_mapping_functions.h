/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef VIDEO_CORRUPTION_DETECTION_GENERIC_MAPPING_FUNCTIONS_H_
#define VIDEO_CORRUPTION_DETECTION_GENERIC_MAPPING_FUNCTIONS_H_

#include "api/video/video_codec_type.h"

namespace webrtc {

struct FilterSettings {
  double std_dev = 0.0;
  int luma_error_threshold = 0;
  int chroma_error_threshold = 0;
};

FilterSettings GetCorruptionFilterSettings(int qp, VideoCodecType codec_type);

}  // namespace webrtc

#endif  // VIDEO_CORRUPTION_DETECTION_GENERIC_MAPPING_FUNCTIONS_H_
