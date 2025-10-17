/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef COMMON_AUDIO_MOCKS_MOCK_SMOOTHING_FILTER_H_
#define COMMON_AUDIO_MOCKS_MOCK_SMOOTHING_FILTER_H_

#include "common_audio/smoothing_filter.h"
#include "test/gmock.h"

namespace webrtc {

class MockSmoothingFilter : public SmoothingFilter {
 public:
  MOCK_METHOD(void, AddSample, (float), (override));
  MOCK_METHOD(std::optional<float>, GetAverage, (), (override));
  MOCK_METHOD(bool, SetTimeConstantMs, (int), (override));
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_MOCKS_MOCK_SMOOTHING_FILTER_H_
