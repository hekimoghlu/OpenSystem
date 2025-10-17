/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_ECHO_CONTROL_MOCK_H_
#define MODULES_AUDIO_PROCESSING_TEST_ECHO_CONTROL_MOCK_H_

#include "api/audio/echo_control.h"
#include "test/gmock.h"

namespace webrtc {

class AudioBuffer;

class MockEchoControl : public EchoControl {
 public:
  MOCK_METHOD(void, AnalyzeRender, (AudioBuffer * render), (override));
  MOCK_METHOD(void, AnalyzeCapture, (AudioBuffer * capture), (override));
  MOCK_METHOD(void,
              ProcessCapture,
              (AudioBuffer * capture, bool echo_path_change),
              (override));
  MOCK_METHOD(void,
              ProcessCapture,
              (AudioBuffer * capture,
               AudioBuffer* linear_output,
               bool echo_path_change),
              (override));
  MOCK_METHOD(EchoControl::Metrics, GetMetrics, (), (const, override));
  MOCK_METHOD(void, SetAudioBufferDelay, (int delay_ms), (override));
  MOCK_METHOD(void,
              SetCaptureOutputUsage,
              (bool capture_output_used),
              (override));
  MOCK_METHOD(bool, ActiveProcessing, (), (const, override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_ECHO_CONTROL_MOCK_H_
