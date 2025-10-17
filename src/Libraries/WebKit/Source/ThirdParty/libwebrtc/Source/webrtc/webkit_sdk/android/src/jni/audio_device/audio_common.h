/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_AUDIO_DEVICE_AUDIO_COMMON_H_
#define SDK_ANDROID_SRC_JNI_AUDIO_DEVICE_AUDIO_COMMON_H_

namespace webrtc {

namespace jni {

const int kDefaultSampleRate = 44100;
// Delay estimates for the two different supported modes. These values are based
// on real-time round-trip delay estimates on a large set of devices and they
// are lower bounds since the filter length is 128 ms, so the AEC works for
// delays in the range [50, ~170] ms and [150, ~270] ms. Note that, in most
// cases, the lowest delay estimate will not be utilized since devices that
// support low-latency output audio often supports HW AEC as well.
const int kLowLatencyModeDelayEstimateInMilliseconds = 50;
const int kHighLatencyModeDelayEstimateInMilliseconds = 150;

}  // namespace jni

}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_AUDIO_DEVICE_AUDIO_COMMON_H_
