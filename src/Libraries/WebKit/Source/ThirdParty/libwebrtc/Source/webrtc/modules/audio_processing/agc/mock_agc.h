/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC_MOCK_AGC_H_
#define MODULES_AUDIO_PROCESSING_AGC_MOCK_AGC_H_

#include "api/array_view.h"
#include "modules/audio_processing/agc/agc.h"
#include "test/gmock.h"

namespace webrtc {

class MockAgc : public Agc {
 public:
  virtual ~MockAgc() {}
  MOCK_METHOD(void, Process, (rtc::ArrayView<const int16_t> audio), (override));
  MOCK_METHOD(bool, GetRmsErrorDb, (int* error), (override));
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(int, set_target_level_dbfs, (int level), (override));
  MOCK_METHOD(int, target_level_dbfs, (), (const, override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC_MOCK_AGC_H_
