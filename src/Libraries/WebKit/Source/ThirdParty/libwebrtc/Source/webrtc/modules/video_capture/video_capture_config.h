/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_CONFIG_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_CONFIG_H_

namespace webrtc {
namespace videocapturemodule {
enum { kDefaultWidth = 640 };     // Start width
enum { kDefaultHeight = 480 };    // Start heigt
enum { kDefaultFrameRate = 30 };  // Start frame rate

enum { kMaxFrameRate = 60 };  // Max allowed frame rate of the start image

enum { kDefaultCaptureDelay = 120 };
enum {
  kMaxCaptureDelay = 270
};  // Max capture delay allowed in the precompiled capture delay values.

enum { kFrameRateCallbackInterval = 1000 };
enum { kFrameRateCountHistorySize = 90 };
enum { kFrameRateHistoryWindowMs = 2000 };
}  // namespace videocapturemodule
}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_CONFIG_H_
