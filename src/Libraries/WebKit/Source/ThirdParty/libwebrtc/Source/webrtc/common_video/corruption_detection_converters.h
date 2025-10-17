/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#ifndef COMMON_VIDEO_CORRUPTION_DETECTION_CONVERTERS_H_
#define COMMON_VIDEO_CORRUPTION_DETECTION_CONVERTERS_H_

#include <optional>

#include "common_video/corruption_detection_message.h"
#include "common_video/frame_instrumentation_data.h"

namespace webrtc {

std::optional<FrameInstrumentationData>
ConvertCorruptionDetectionMessageToFrameInstrumentationData(
    const CorruptionDetectionMessage& message,
    int previous_sequence_index);
std::optional<FrameInstrumentationSyncData>
ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(
    const CorruptionDetectionMessage& message,
    int previous_sequence_index);
std::optional<CorruptionDetectionMessage>
ConvertFrameInstrumentationDataToCorruptionDetectionMessage(
    const FrameInstrumentationData& frame_instrumentation_data);
std::optional<CorruptionDetectionMessage>
ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(
    const FrameInstrumentationSyncData& frame_instrumentation_sync_data);
}  // namespace webrtc

#endif  // COMMON_VIDEO_CORRUPTION_DETECTION_CONVERTERS_H_
