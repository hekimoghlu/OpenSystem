/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_EXPAND_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_EXPAND_H_

#include "modules/audio_coding/neteq/expand.h"
#include "test/gmock.h"

namespace webrtc {

class MockExpand : public Expand {
 public:
  MockExpand(BackgroundNoise* background_noise,
             SyncBuffer* sync_buffer,
             RandomVector* random_vector,
             StatisticsCalculator* statistics,
             int fs,
             size_t num_channels)
      : Expand(background_noise,
               sync_buffer,
               random_vector,
               statistics,
               fs,
               num_channels) {}
  ~MockExpand() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(int, Process, (AudioMultiVector * output), (override));
  MOCK_METHOD(void, SetParametersForNormalAfterExpand, (), (override));
  MOCK_METHOD(void, SetParametersForMergeAfterExpand, (), (override));
  MOCK_METHOD(size_t, overlap_length, (), (const, override));
};

}  // namespace webrtc

namespace webrtc {

class MockExpandFactory : public ExpandFactory {
 public:
  MOCK_METHOD(Expand*,
              Create,
              (BackgroundNoise * background_noise,
               SyncBuffer* sync_buffer,
               RandomVector* random_vector,
               StatisticsCalculator* statistics,
               int fs,
               size_t num_channels),
              (const, override));
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_EXPAND_H_
