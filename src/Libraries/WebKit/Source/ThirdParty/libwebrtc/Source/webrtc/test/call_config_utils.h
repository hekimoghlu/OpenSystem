/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#ifndef TEST_CALL_CONFIG_UTILS_H_
#define TEST_CALL_CONFIG_UTILS_H_

#include "call/video_receive_stream.h"
#include "rtc_base/strings/json.h"

namespace webrtc {
namespace test {

// Deserializes a JSON representation of the VideoReceiveStreamInterface::Config
// back into a valid object. This will not initialize the decoders or the
// renderer.
VideoReceiveStreamInterface::Config ParseVideoReceiveStreamJsonConfig(
    webrtc::Transport* transport,
    const Json::Value& json);

// Serialize a VideoReceiveStreamInterface::Config into a Json object.
Json::Value GenerateVideoReceiveStreamJsonConfig(
    const VideoReceiveStreamInterface::Config& config);

}  // namespace test
}  // namespace webrtc

#endif  // TEST_CALL_CONFIG_UTILS_H_
