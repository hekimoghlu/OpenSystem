/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_INCLUDE_MOCK_AUDIO_PROCESSING_H_
#define MODULES_AUDIO_PROCESSING_INCLUDE_MOCK_AUDIO_PROCESSING_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "api/audio/audio_processing.h"
#include "api/audio/audio_processing_statistics.h"
#include "api/environment/environment.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_base.h"
#include "modules/audio_processing/include/aec_dump.h"
#include "test/gmock.h"

namespace webrtc {

namespace test {
class MockCustomProcessing : public CustomProcessing {
 public:
  virtual ~MockCustomProcessing() {}
  MOCK_METHOD(void,
              Initialize,
              (int sample_rate_hz, int num_channels),
              (override));
  MOCK_METHOD(void, Process, (AudioBuffer * audio), (override));
  MOCK_METHOD(void,
              SetRuntimeSetting,
              (AudioProcessing::RuntimeSetting setting),
              (override));
  MOCK_METHOD(std::string, ToString, (), (const, override));
};

class MockCustomAudioAnalyzer : public CustomAudioAnalyzer {
 public:
  virtual ~MockCustomAudioAnalyzer() {}
  MOCK_METHOD(void,
              Initialize,
              (int sample_rate_hz, int num_channels),
              (override));
  MOCK_METHOD(void, Analyze, (const AudioBuffer* audio), (override));
  MOCK_METHOD(std::string, ToString, (), (const, override));
};

class MockEchoControl : public EchoControl {
 public:
  virtual ~MockEchoControl() {}
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
  MOCK_METHOD(Metrics, GetMetrics, (), (const, override));
  MOCK_METHOD(void, SetAudioBufferDelay, (int delay_ms), (override));
  MOCK_METHOD(bool, ActiveProcessing, (), (const, override));
};

class MockEchoDetector : public EchoDetector {
 public:
  virtual ~MockEchoDetector() {}
  MOCK_METHOD(void,
              Initialize,
              (int capture_sample_rate_hz,
               int num_capture_channels,
               int render_sample_rate_hz,
               int num_render_channels),
              (override));
  MOCK_METHOD(void,
              AnalyzeRenderAudio,
              (rtc::ArrayView<const float> render_audio),
              (override));
  MOCK_METHOD(void,
              AnalyzeCaptureAudio,
              (rtc::ArrayView<const float> capture_audio),
              (override));
  MOCK_METHOD(Metrics, GetMetrics, (), (const, override));
};

class MockAudioProcessing : public AudioProcessing {
 public:
  MockAudioProcessing() {}

  virtual ~MockAudioProcessing() {}

  MOCK_METHOD(int, Initialize, (), (override));
  MOCK_METHOD(int,
              Initialize,
              (const ProcessingConfig& processing_config),
              (override));
  MOCK_METHOD(void, ApplyConfig, (const Config& config), (override));
  MOCK_METHOD(int, proc_sample_rate_hz, (), (const, override));
  MOCK_METHOD(int, proc_split_sample_rate_hz, (), (const, override));
  MOCK_METHOD(size_t, num_input_channels, (), (const, override));
  MOCK_METHOD(size_t, num_proc_channels, (), (const, override));
  MOCK_METHOD(size_t, num_output_channels, (), (const, override));
  MOCK_METHOD(size_t, num_reverse_channels, (), (const, override));
  MOCK_METHOD(void, set_output_will_be_muted, (bool muted), (override));
  MOCK_METHOD(void, SetRuntimeSetting, (RuntimeSetting setting), (override));
  MOCK_METHOD(bool, PostRuntimeSetting, (RuntimeSetting setting), (override));
  MOCK_METHOD(int,
              ProcessStream,
              (const int16_t* const src,
               const StreamConfig& input_config,
               const StreamConfig& output_config,
               int16_t* const dest),
              (override));
  MOCK_METHOD(int,
              ProcessStream,
              (const float* const* src,
               const StreamConfig& input_config,
               const StreamConfig& output_config,
               float* const* dest),
              (override));
  MOCK_METHOD(int,
              ProcessReverseStream,
              (const int16_t* const src,
               const StreamConfig& input_config,
               const StreamConfig& output_config,
               int16_t* const dest),
              (override));
  MOCK_METHOD(int,
              AnalyzeReverseStream,
              (const float* const* data, const StreamConfig& reverse_config),
              (override));
  MOCK_METHOD(int,
              ProcessReverseStream,
              (const float* const* src,
               const StreamConfig& input_config,
               const StreamConfig& output_config,
               float* const* dest),
              (override));
  MOCK_METHOD(bool,
              GetLinearAecOutput,
              ((rtc::ArrayView<std::array<float, 160>> linear_output)),
              (const, override));
  MOCK_METHOD(int, set_stream_delay_ms, (int delay), (override));
  MOCK_METHOD(int, stream_delay_ms, (), (const, override));
  MOCK_METHOD(void, set_stream_key_pressed, (bool key_pressed), (override));
  MOCK_METHOD(void, set_stream_analog_level, (int), (override));
  MOCK_METHOD(int, recommended_stream_analog_level, (), (const, override));
  MOCK_METHOD(bool,
              CreateAndAttachAecDump,
              (absl::string_view file_name,
               int64_t max_log_size_bytes,
               absl::Nonnull<TaskQueueBase*> worker_queue),
              (override));
  MOCK_METHOD(bool,
              CreateAndAttachAecDump,
              (FILE * handle,
               int64_t max_log_size_bytes,
               absl::Nonnull<TaskQueueBase*> worker_queue),
              (override));
  MOCK_METHOD(void, AttachAecDump, (std::unique_ptr<AecDump>), (override));
  MOCK_METHOD(void, DetachAecDump, (), (override));

  MOCK_METHOD(AudioProcessingStats, GetStatistics, (), (override));
  MOCK_METHOD(AudioProcessingStats, GetStatistics, (bool), (override));

  MOCK_METHOD(AudioProcessing::Config, GetConfig, (), (const, override));
};

class MockAudioProcessingBuilder : public AudioProcessingBuilderInterface {
 public:
  MOCK_METHOD(scoped_refptr<AudioProcessing>,
              Build,
              (const Environment&),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_INCLUDE_MOCK_AUDIO_PROCESSING_H_
