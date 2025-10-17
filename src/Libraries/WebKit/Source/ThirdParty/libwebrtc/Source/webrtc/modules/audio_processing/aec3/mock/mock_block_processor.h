/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_BLOCK_PROCESSOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_BLOCK_PROCESSOR_H_

#include <vector>

#include "modules/audio_processing/aec3/block_processor.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockBlockProcessor : public BlockProcessor {
 public:
  MockBlockProcessor();
  virtual ~MockBlockProcessor();

  MOCK_METHOD(void,
              ProcessCapture,
              (bool level_change,
               bool saturated_microphone_signal,
               Block* linear_output,
               Block* capture_block),
              (override));
  MOCK_METHOD(void, BufferRender, (const Block& block), (override));
  MOCK_METHOD(void,
              UpdateEchoLeakageStatus,
              (bool leakage_detected),
              (override));
  MOCK_METHOD(void,
              GetMetrics,
              (EchoControl::Metrics * metrics),
              (const, override));
  MOCK_METHOD(void, SetAudioBufferDelay, (int delay_ms), (override));
  MOCK_METHOD(void,
              SetCaptureOutputUsage,
              (bool capture_output_used),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_BLOCK_PROCESSOR_H_
