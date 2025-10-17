/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC_DUMP_CAPTURE_STREAM_INFO_H_
#define MODULES_AUDIO_PROCESSING_AEC_DUMP_CAPTURE_STREAM_INFO_H_

#include <memory>
#include <utility>

#include "modules/audio_processing/include/aec_dump.h"

// Files generated at build-time by the protobuf compiler.
#ifdef WEBRTC_ANDROID_PLATFORM_BUILD
#include "external/webrtc/webrtc/modules/audio_processing/debug.pb.h"
#else
#include "modules/audio_processing/debug.pb.h"
#endif

namespace webrtc {

class CaptureStreamInfo {
 public:
  CaptureStreamInfo() { CreateNewEvent(); }
  CaptureStreamInfo(const CaptureStreamInfo&) = delete;
  CaptureStreamInfo& operator=(const CaptureStreamInfo&) = delete;
  ~CaptureStreamInfo() = default;

  void AddInput(const AudioFrameView<const float>& src);
  void AddOutput(const AudioFrameView<const float>& src);

  void AddInput(const int16_t* const data,
                int num_channels,
                int samples_per_channel);
  void AddOutput(const int16_t* const data,
                 int num_channels,
                 int samples_per_channel);

  void AddAudioProcessingState(const AecDump::AudioProcessingState& state);

  std::unique_ptr<audioproc::Event> FetchEvent() {
    std::unique_ptr<audioproc::Event> result = std::move(event_);
    CreateNewEvent();
    return result;
  }

 private:
  void CreateNewEvent() {
    event_ = std::make_unique<audioproc::Event>();
    event_->set_type(audioproc::Event::STREAM);
  }
  std::unique_ptr<audioproc::Event> event_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC_DUMP_CAPTURE_STREAM_INFO_H_
