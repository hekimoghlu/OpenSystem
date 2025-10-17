/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_REFERENCE_LESS_VIDEO_ANALYSIS_LIB_H_
#define RTC_TOOLS_FRAME_ANALYZER_REFERENCE_LESS_VIDEO_ANALYSIS_LIB_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "api/scoped_refptr.h"
#include "rtc_tools/video_file_reader.h"

// Returns true if the frame is frozen based on psnr and ssim freezing
// threshold values.
bool frozen_frame(std::vector<double> psnr_per_frame,
                  std::vector<double> ssim_per_frame,
                  size_t frame);

// Returns the vector of identical cluster of frames that are frozen
// and appears continuously.
std::vector<int> find_frame_clusters(const std::vector<double>& psnr_per_frame,
                                     const std::vector<double>& ssim_per_frame);

// Prints various freezing metrics like identical frames,
// total unique frames etc.
void print_freezing_metrics(const std::vector<double>& psnr_per_frame,
                            const std::vector<double>& ssim_per_frame);

// Compute the metrics like freezing score based on PSNR and SSIM values for a
// given video file.
void compute_metrics(const rtc::scoped_refptr<webrtc::test::Video>& video,
                     std::vector<double>* psnr_per_frame,
                     std::vector<double>* ssim_per_frame);

// Compute freezing score metrics and prints the metrics
// for a list of video files.
int run_analysis(const std::string& video_file);

#endif  // RTC_TOOLS_FRAME_ANALYZER_REFERENCE_LESS_VIDEO_ANALYSIS_LIB_H_
