/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#ifndef RTC_TOOLS_VIDEO_FILE_WRITER_H_
#define RTC_TOOLS_VIDEO_FILE_WRITER_H_

#include <string>

#include "api/scoped_refptr.h"
#include "rtc_tools/video_file_reader.h"

namespace webrtc {
namespace test {

// Writes video to file, determining YUV or Y4M format from the file extension.
void WriteVideoToFile(const rtc::scoped_refptr<Video>& video,
                      const std::string& file_name,
                      int fps);

// Writes Y4M video to file.
void WriteY4mVideoToFile(const rtc::scoped_refptr<Video>& video,
                         const std::string& file_name,
                         int fps);

// Writes YUV video to file.
void WriteYuvVideoToFile(const rtc::scoped_refptr<Video>& video,
                         const std::string& file_name,
                         int fps);

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_VIDEO_FILE_WRITER_H_
