/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_VIDEO_GEOMETRY_ALIGNER_H_
#define RTC_TOOLS_FRAME_ANALYZER_VIDEO_GEOMETRY_ALIGNER_H_

#include "api/video/video_frame_buffer.h"
#include "rtc_tools/video_file_reader.h"

namespace webrtc {
namespace test {

struct CropRegion {
  // Each value represents how much to crop from each side. Left is where x=0,
  // and top is where y=0. All values equal to zero represents no cropping.
  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;
};

// Crops and zooms in on the cropped region so that the returned frame has the
// same resolution as the input frame.
rtc::scoped_refptr<I420BufferInterface> CropAndZoom(
    const CropRegion& crop_region,
    const rtc::scoped_refptr<I420BufferInterface>& frame);

// Calculate the optimal cropping region on the reference frame to maximize SSIM
// to the test frame.
CropRegion CalculateCropRegion(
    const rtc::scoped_refptr<I420BufferInterface>& reference_frame,
    const rtc::scoped_refptr<I420BufferInterface>& test_frame);

// Returns a cropped and zoomed version of the reference frame that matches up
// to the test frame. This is a simple helper function on top of
// CalculateCropRegion() and CropAndZoom().
rtc::scoped_refptr<I420BufferInterface> AdjustCropping(
    const rtc::scoped_refptr<I420BufferInterface>& reference_frame,
    const rtc::scoped_refptr<I420BufferInterface>& test_frame);

// Returns a cropped and zoomed version of the reference video that matches up
// to the test video. Frames are individually adjusted for cropping.
rtc::scoped_refptr<Video> AdjustCropping(
    const rtc::scoped_refptr<Video>& reference_video,
    const rtc::scoped_refptr<Video>& test_video);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_FRAME_ANALYZER_VIDEO_GEOMETRY_ALIGNER_H_
