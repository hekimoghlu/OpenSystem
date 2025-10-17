/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_VIDEO_COLOR_ALIGNER_H_
#define RTC_TOOLS_FRAME_ANALYZER_VIDEO_COLOR_ALIGNER_H_

#include <array>

#include "api/scoped_refptr.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_tools/video_file_reader.h"

namespace webrtc {
namespace test {

// Represents a linear color transformation from [y, u, v] to [y', u', v']
// through the equation: [y', u', v'] = [y, u, v, 1] * matrix.
using ColorTransformationMatrix = std::array<std::array<float, 4>, 3>;

// Calculate the optimal color transformation that should be applied to the test
// video to match as closely as possible to the reference video.
ColorTransformationMatrix CalculateColorTransformationMatrix(
    const rtc::scoped_refptr<Video>& reference_video,
    const rtc::scoped_refptr<Video>& test_video);

// Calculate color transformation for a single I420 frame.
ColorTransformationMatrix CalculateColorTransformationMatrix(
    const rtc::scoped_refptr<I420BufferInterface>& reference_frame,
    const rtc::scoped_refptr<I420BufferInterface>& test_frame);

// Apply a color transformation to a video.
rtc::scoped_refptr<Video> AdjustColors(
    const ColorTransformationMatrix& color_matrix,
    const rtc::scoped_refptr<Video>& video);

// Apply a color transformation to a single I420 frame.
rtc::scoped_refptr<I420BufferInterface> AdjustColors(
    const ColorTransformationMatrix& color_matrix,
    const rtc::scoped_refptr<I420BufferInterface>& frame);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_FRAME_ANALYZER_VIDEO_COLOR_ALIGNER_H_
