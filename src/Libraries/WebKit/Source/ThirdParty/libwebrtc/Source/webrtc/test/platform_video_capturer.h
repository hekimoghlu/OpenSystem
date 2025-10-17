/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#ifndef TEST_PLATFORM_VIDEO_CAPTURER_H_
#define TEST_PLATFORM_VIDEO_CAPTURER_H_

#include <memory>

#include "test/test_video_capturer.h"

namespace webrtc {
namespace test {

std::unique_ptr<TestVideoCapturer> CreateVideoCapturer(
    size_t width,
    size_t height,
    size_t target_fps,
    size_t capture_device_index);

}  // namespace test
}  // namespace webrtc

#endif  // TEST_PLATFORM_VIDEO_CAPTURER_H_
