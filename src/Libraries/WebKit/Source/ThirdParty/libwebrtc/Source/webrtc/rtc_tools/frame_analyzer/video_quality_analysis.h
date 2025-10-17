/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_VIDEO_QUALITY_ANALYSIS_H_
#define RTC_TOOLS_FRAME_ANALYZER_VIDEO_QUALITY_ANALYSIS_H_

#include <stdio.h>

#include <string>
#include <vector>

#include "api/scoped_refptr.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_tools/video_file_reader.h"

namespace webrtc {
namespace test {

struct AnalysisResult {
  AnalysisResult() {}
  AnalysisResult(int frame_number, double psnr_value, double ssim_value)
      : frame_number(frame_number),
        psnr_value(psnr_value),
        ssim_value(ssim_value) {}
  int frame_number;
  double psnr_value;
  double ssim_value;
};

struct ResultsContainer {
  ResultsContainer();
  ~ResultsContainer();

  std::vector<AnalysisResult> frames;
  int max_repeated_frames = 0;
  int max_skipped_frames = 0;
  int total_skipped_frames = 0;
  int decode_errors_ref = 0;
  int decode_errors_test = 0;
};

// A function to run the PSNR and SSIM analysis on the test file. The test file
// comprises the frames that were captured during the quality measurement test.
// There may be missing or duplicate frames. Also the frames start at a random
// position in the original video. We also need to provide a map from test frame
// indices to reference frame indices.
std::vector<AnalysisResult> RunAnalysis(
    const rtc::scoped_refptr<webrtc::test::Video>& reference_video,
    const rtc::scoped_refptr<webrtc::test::Video>& test_video,
    const std::vector<size_t>& test_frame_indices);

// Compute PSNR for an I420 buffer (all planes). The max return value (in the
// case where the test and reference frames are exactly the same) will be 48.
double Psnr(const rtc::scoped_refptr<I420BufferInterface>& ref_buffer,
            const rtc::scoped_refptr<I420BufferInterface>& test_buffer);

// Compute SSIM for an I420 buffer (all planes). The max return value (in the
// case where the test and reference frames are exactly the same) will be 1.
double Ssim(const rtc::scoped_refptr<I420BufferInterface>& ref_buffer,
            const rtc::scoped_refptr<I420BufferInterface>& test_buffer);

// Prints the result from the analysis in Chromium performance
// numbers compatible format to stdout. If the results object contains no frames
// no output will be written.
void PrintAnalysisResults(const std::string& label,
                          ResultsContainer& results,
                          MetricsLogger& logger);

struct Cluster {
  // Corresponding reference frame index for this cluster.
  size_t index;
  // The number of sequential frames that mapped to the same reference frame
  // index.
  int number_of_repeated_frames;
};

// Clusters sequentially repeated frames. For example, the sequence {100, 102,
// 102, 103} will be mapped to {{100, 1}, {102, 2}, {103, 1}}.
std::vector<Cluster> CalculateFrameClusters(const std::vector<size_t>& indices);

// Get number of max sequentially repeated frames in the test video. This number
// will be one if we only store unique frames in the test video.
int GetMaxRepeatedFrames(const std::vector<Cluster>& clusters);

// Get the longest sequence of skipped reference frames. This corresponds to the
// longest freeze in the test video.
int GetMaxSkippedFrames(const std::vector<Cluster>& clusters);

// Get total number of skipped frames in the test video.
int GetTotalNumberOfSkippedFrames(const std::vector<Cluster>& clusters);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_FRAME_ANALYZER_VIDEO_QUALITY_ANALYSIS_H_
