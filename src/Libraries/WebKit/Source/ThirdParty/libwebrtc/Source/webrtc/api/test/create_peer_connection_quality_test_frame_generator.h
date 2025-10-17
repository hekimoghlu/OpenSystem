/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#ifndef API_TEST_CREATE_PEER_CONNECTION_QUALITY_TEST_FRAME_GENERATOR_H_
#define API_TEST_CREATE_PEER_CONNECTION_QUALITY_TEST_FRAME_GENERATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "api/test/frame_generator_interface.h"
#include "api/test/pclf/media_configuration.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Creates a frame generator that produces frames with small squares that move
// randomly towards the lower right corner. `type` has the default value
// FrameGeneratorInterface::OutputType::I420. video_config specifies frame
// weight and height.
std::unique_ptr<test::FrameGeneratorInterface> CreateSquareFrameGenerator(
    const VideoConfig& video_config,
    std::optional<test::FrameGeneratorInterface::OutputType> type);

// Creates a frame generator that plays frames from the yuv file.
std::unique_ptr<test::FrameGeneratorInterface> CreateFromYuvFileFrameGenerator(
    const VideoConfig& video_config,
    std::string filename);

// Creates a proper frame generator for testing screen sharing.
std::unique_ptr<test::FrameGeneratorInterface> CreateScreenShareFrameGenerator(
    const VideoConfig& video_config,
    const ScreenShareConfig& screen_share_config);

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // API_TEST_CREATE_PEER_CONNECTION_QUALITY_TEST_FRAME_GENERATOR_H_
