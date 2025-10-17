/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_SIMULCAST_DUMMY_BUFFER_HELPER_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_SIMULCAST_DUMMY_BUFFER_HELPER_H_

#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Creates a special video frame buffer that should be used to create frames
// during Selective Forwarding Unit (SFU) emulation. Such frames are used when
// original was discarded and some frame is required to be passed upstream
// to make WebRTC pipeline happy and not request key frame on the received
// stream due to lack of incoming frames.
rtc::scoped_refptr<webrtc::VideoFrameBuffer> CreateDummyFrameBuffer();

// Tests if provided frame contains a buffer created by
// `CreateDummyFrameBuffer`.
bool IsDummyFrame(const webrtc::VideoFrame& video_frame);

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_SIMULCAST_DUMMY_BUFFER_HELPER_H_
