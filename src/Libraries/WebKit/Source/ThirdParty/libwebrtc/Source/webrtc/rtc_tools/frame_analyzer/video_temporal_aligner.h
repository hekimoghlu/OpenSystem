/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_VIDEO_TEMPORAL_ALIGNER_H_
#define RTC_TOOLS_FRAME_ANALYZER_VIDEO_TEMPORAL_ALIGNER_H_

#include <stddef.h>

#include <vector>

#include "api/scoped_refptr.h"
#include "rtc_tools/video_file_reader.h"

namespace webrtc {
namespace test {

// Returns a vector with the same size as the given test video. Each index
// corresponds to what reference frame that test frame matches to. These
// indices are strictly increasing and might loop around the reference video,
// e.g. their values can be bigger than the number of frames in the reference
// video and they should be interpreted modulo that size. The matching frames
// will be determined by maximizing SSIM.
std::vector<size_t> FindMatchingFrameIndices(
    const rtc::scoped_refptr<Video>& reference_video,
    const rtc::scoped_refptr<Video>& test_video);

// Generate a new video using the frames from the original video. The returned
// video will have the same number of frames as the size of `indices`, and
// frame nr i in the returned video will point to frame nr indices[i] in the
// original video.
rtc::scoped_refptr<Video> ReorderVideo(const rtc::scoped_refptr<Video>& video,
                                       const std::vector<size_t>& indices);

// Returns a modified version of the reference video where the frames have
// been aligned to the test video. The test video is assumed to be captured
// during a quality measurement test where the reference video is the source.
// The test video may start at an arbitrary position in the reference video
// and there might be missing frames. The reference video is assumed to loop
// over when it reaches the end. The returned result is a version of the
// reference video where the missing frames are left out so it aligns to the
// test video.
rtc::scoped_refptr<Video> GenerateAlignedReferenceVideo(
    const rtc::scoped_refptr<Video>& reference_video,
    const rtc::scoped_refptr<Video>& test_video);

// As above, but using precalculated indices.
rtc::scoped_refptr<Video> GenerateAlignedReferenceVideo(
    const rtc::scoped_refptr<Video>& reference_video,
    const std::vector<size_t>& indices);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_FRAME_ANALYZER_VIDEO_TEMPORAL_ALIGNER_H_
