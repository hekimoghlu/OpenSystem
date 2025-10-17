/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#ifndef API_FRAME_TRANSFORMER_FACTORY_H_
#define API_FRAME_TRANSFORMER_FACTORY_H_

#include <memory>

#include "api/frame_transformer_interface.h"
#include "rtc_base/system/rtc_export.h"

// This file contains EXPERIMENTAL functions to create video frames from
// either an old video frame or directly from parameters.
// These functions will be used in Chrome functionality to manipulate
// encoded frames from Javascript.
namespace webrtc {

// TODO(bugs.webrtc.org/14708): Add the required parameters to these APIs.
std::unique_ptr<TransformableVideoFrameInterface> CreateVideoSenderFrame();
// TODO(bugs.webrtc.org/14708): Consider whether Receiver frames ever make sense
// to create.
std::unique_ptr<TransformableVideoFrameInterface> CreateVideoReceiverFrame();
// Creates a new frame with the same metadata as the original.
// The original can be a sender or receiver frame.
RTC_EXPORT std::unique_ptr<TransformableAudioFrameInterface> CloneAudioFrame(
    TransformableAudioFrameInterface* original);
RTC_EXPORT std::unique_ptr<TransformableVideoFrameInterface> CloneVideoFrame(
    TransformableVideoFrameInterface* original);
}  // namespace webrtc

#endif  // API_FRAME_TRANSFORMER_FACTORY_H_
