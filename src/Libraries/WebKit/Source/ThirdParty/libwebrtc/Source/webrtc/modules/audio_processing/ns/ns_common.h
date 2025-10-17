/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_NS_COMMON_H_
#define MODULES_AUDIO_PROCESSING_NS_NS_COMMON_H_

#include <cstddef>

namespace webrtc {

constexpr size_t kFftSize = 256;
constexpr size_t kFftSizeBy2Plus1 = kFftSize / 2 + 1;
constexpr size_t kNsFrameSize = 160;
constexpr size_t kOverlapSize = kFftSize - kNsFrameSize;

constexpr int kShortStartupPhaseBlocks = 50;
constexpr int kLongStartupPhaseBlocks = 200;
constexpr int kFeatureUpdateWindowSize = 500;

constexpr float kLtrFeatureThr = 0.5f;
constexpr float kBinSizeLrt = 0.1f;
constexpr float kBinSizeSpecFlat = 0.05f;
constexpr float kBinSizeSpecDiff = 0.1f;

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_NS_COMMON_H_
