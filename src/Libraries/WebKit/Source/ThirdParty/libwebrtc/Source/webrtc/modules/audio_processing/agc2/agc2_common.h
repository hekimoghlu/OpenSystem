/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_AGC2_COMMON_H_
#define MODULES_AUDIO_PROCESSING_AGC2_AGC2_COMMON_H_

namespace webrtc {

constexpr float kMinFloatS16Value = -32768.0f;
constexpr float kMaxFloatS16Value = 32767.0f;
constexpr float kMaxAbsFloatS16Value = 32768.0f;

// Minimum audio level in dBFS scale for S16 samples.
constexpr float kMinLevelDbfs = -90.31f;

constexpr int kFrameDurationMs = 10;
constexpr int kSubFramesInFrame = 20;
constexpr int kMaximalNumberOfSamplesPerChannel = 480;

// Adaptive digital gain applier settings.

// At what limiter levels should we start decreasing the adaptive digital gain.
constexpr float kLimiterThresholdForAgcGainDbfs = -1.0f;

// Number of milliseconds to wait to periodically reset the VAD.
constexpr int kVadResetPeriodMs = 1500;

// Speech probability threshold to detect speech activity.
constexpr float kVadConfidenceThreshold = 0.95f;

// Minimum number of adjacent speech frames having a sufficiently high speech
// probability to reliably detect speech activity.
constexpr int kAdjacentSpeechFramesThreshold = 12;

// Number of milliseconds of speech frames to observe to make the estimator
// confident.
constexpr float kLevelEstimatorTimeToConfidenceMs = 400;
constexpr float kLevelEstimatorLeakFactor =
    1.0f - 1.0f / kLevelEstimatorTimeToConfidenceMs;

// Saturation Protector settings.
constexpr float kSaturationProtectorInitialHeadroomDb = 20.0f;
constexpr int kSaturationProtectorBufferSize = 4;

// Number of interpolation points for each region of the limiter.
// These values have been tuned to limit the interpolated gain curve error given
// the limiter parameters and allowing a maximum error of +/- 32768^-1.
constexpr int kInterpolatedGainCurveKneePoints = 22;
constexpr int kInterpolatedGainCurveBeyondKneePoints = 10;
constexpr int kInterpolatedGainCurveTotalPoints =
    kInterpolatedGainCurveKneePoints + kInterpolatedGainCurveBeyondKneePoints;

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_AGC2_COMMON_H_
