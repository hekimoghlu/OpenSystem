/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_IMPL_H_
#define MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "api/task_queue/task_queue_base.h"
#include "modules/audio_processing/aec_dump/capture_stream_info.h"
#include "modules/audio_processing/include/aec_dump.h"
#include "rtc_base/race_checker.h"
#include "rtc_base/system/file_wrapper.h"
#include "rtc_base/thread_annotations.h"

// Files generated at build-time by the protobuf compiler.
#ifdef WEBRTC_ANDROID_PLATFORM_BUILD
#include "external/webrtc/webrtc/modules/audio_processing/debug.pb.h"
#else
#include "modules/audio_processing/debug.pb.h"
#endif

namespace webrtc {

// Task-queue based implementation of AecDump. It is thread safe by
// relying on locks in TaskQueue.
class AecDumpImpl : public AecDump {
 public:
  // `max_log_size_bytes` - maximum number of bytes to write to the debug file,
  // `max_log_size_bytes == -1` means the log size will be unlimited.
  AecDumpImpl(FileWrapper debug_file,
              int64_t max_log_size_bytes,
              absl::Nonnull<TaskQueueBase*> worker_queue);
  AecDumpImpl(const AecDumpImpl&) = delete;
  AecDumpImpl& operator=(const AecDumpImpl&) = delete;
  ~AecDumpImpl() override;

  void WriteInitMessage(const ProcessingConfig& api_format,
                        int64_t time_now_ms) override;
  void AddCaptureStreamInput(const AudioFrameView<const float>& src) override;
  void AddCaptureStreamOutput(const AudioFrameView<const float>& src) override;
  void AddCaptureStreamInput(const int16_t* const data,
                             int num_channels,
                             int samples_per_channel) override;
  void AddCaptureStreamOutput(const int16_t* const data,
                              int num_channels,
                              int samples_per_channel) override;
  void AddAudioProcessingState(const AudioProcessingState& state) override;
  void WriteCaptureStreamMessage() override;

  void WriteRenderStreamMessage(const int16_t* const data,
                                int num_channels,
                                int samples_per_channel) override;
  void WriteRenderStreamMessage(
      const AudioFrameView<const float>& src) override;

  void WriteConfig(const InternalAPMConfig& config) override;

  void WriteRuntimeSetting(
      const AudioProcessing::RuntimeSetting& runtime_setting) override;

 private:
  void PostWriteToFileTask(std::unique_ptr<audioproc::Event> event);

  FileWrapper debug_file_;
  int64_t num_bytes_left_for_log_ = 0;
  rtc::RaceChecker race_checker_;
  absl::Nonnull<TaskQueueBase*> worker_queue_;
  CaptureStreamInfo capture_stream_info_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_IMPL_H_
