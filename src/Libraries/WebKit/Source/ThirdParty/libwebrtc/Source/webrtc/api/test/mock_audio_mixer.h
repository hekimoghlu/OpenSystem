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
#ifndef API_TEST_MOCK_AUDIO_MIXER_H_
#define API_TEST_MOCK_AUDIO_MIXER_H_

#include <cstddef>

#include "api/audio/audio_frame.h"
#include "api/audio/audio_mixer.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockAudioMixer : public AudioMixer {
 public:
  MOCK_METHOD(bool, AddSource, (Source*), (override));
  MOCK_METHOD(void, RemoveSource, (Source*), (override));
  MOCK_METHOD(void, Mix, (size_t number_of_channels, AudioFrame*), (override));
};
}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_MOCK_AUDIO_MIXER_H_
