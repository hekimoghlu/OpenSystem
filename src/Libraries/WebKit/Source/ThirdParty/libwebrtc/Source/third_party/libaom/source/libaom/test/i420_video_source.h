/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#ifndef AOM_TEST_I420_VIDEO_SOURCE_H_
#define AOM_TEST_I420_VIDEO_SOURCE_H_
#include <cstdio>
#include <cstdlib>
#include <string>

#include "test/yuv_video_source.h"

namespace libaom_test {

// This class extends VideoSource to allow parsing of raw yv12
// so that we can do actual file encodes.
class I420VideoSource : public YUVVideoSource {
 public:
  I420VideoSource(const std::string &file_name, unsigned int width,
                  unsigned int height, int rate_numerator, int rate_denominator,
                  unsigned int start, int limit)
      : YUVVideoSource(file_name, AOM_IMG_FMT_I420, width, height,
                       rate_numerator, rate_denominator, start, limit) {}
};

}  // namespace libaom_test

#endif  // AOM_TEST_I420_VIDEO_SOURCE_H_
