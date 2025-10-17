/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_ECHO_REMOVER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_ECHO_REMOVER_H_

#include <optional>
#include <vector>

#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/echo_remover.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockEchoRemover : public EchoRemover {
 public:
  MockEchoRemover();
  virtual ~MockEchoRemover();

  MOCK_METHOD(void,
              ProcessCapture,
              (EchoPathVariability echo_path_variability,
               bool capture_signal_saturation,
               const std::optional<DelayEstimate>& delay_estimate,
               RenderBuffer* render_buffer,
               Block* linear_output,
               Block* capture),
              (override));
  MOCK_METHOD(void,
              UpdateEchoLeakageStatus,
              (bool leakage_detected),
              (override));
  MOCK_METHOD(void,
              GetMetrics,
              (EchoControl::Metrics * metrics),
              (const, override));
  MOCK_METHOD(void,
              SetCaptureOutputUsage,
              (bool capture_output_used),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_ECHO_REMOVER_H_
