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
#ifndef API_AUDIO_BUILTIN_AUDIO_PROCESSING_BUILDER_H_
#define API_AUDIO_BUILTIN_AUDIO_PROCESSING_BUILDER_H_

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/audio/audio_processing.h"
#include "api/audio/echo_control.h"
#include "api/environment/environment.h"
#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT BuiltinAudioProcessingBuilder
    : public AudioProcessingBuilderInterface {
 public:
  BuiltinAudioProcessingBuilder() = default;
  explicit BuiltinAudioProcessingBuilder(const AudioProcessing::Config& config)
      : config_(config) {}
  BuiltinAudioProcessingBuilder(const BuiltinAudioProcessingBuilder&) = delete;
  BuiltinAudioProcessingBuilder& operator=(
      const BuiltinAudioProcessingBuilder&) = delete;
  ~BuiltinAudioProcessingBuilder() override = default;

  // Sets the APM configuration.
  BuiltinAudioProcessingBuilder& SetConfig(
      const AudioProcessing::Config& config) {
    config_ = config;
    return *this;
  }

  // Sets the echo controller factory to inject when APM is created.
  BuiltinAudioProcessingBuilder& SetEchoControlFactory(
      std::unique_ptr<EchoControlFactory> echo_control_factory) {
    echo_control_factory_ = std::move(echo_control_factory);
    return *this;
  }

  // Sets the capture post-processing sub-module to inject when APM is created.
  BuiltinAudioProcessingBuilder& SetCapturePostProcessing(
      std::unique_ptr<CustomProcessing> capture_post_processing) {
    capture_post_processing_ = std::move(capture_post_processing);
    return *this;
  }

  // Sets the render pre-processing sub-module to inject when APM is created.
  BuiltinAudioProcessingBuilder& SetRenderPreProcessing(
      std::unique_ptr<CustomProcessing> render_pre_processing) {
    render_pre_processing_ = std::move(render_pre_processing);
    return *this;
  }

  // Sets the echo detector to inject when APM is created.
  BuiltinAudioProcessingBuilder& SetEchoDetector(
      rtc::scoped_refptr<EchoDetector> echo_detector) {
    echo_detector_ = std::move(echo_detector);
    return *this;
  }

  // Sets the capture analyzer sub-module to inject when APM is created.
  BuiltinAudioProcessingBuilder& SetCaptureAnalyzer(
      std::unique_ptr<CustomAudioAnalyzer> capture_analyzer) {
    capture_analyzer_ = std::move(capture_analyzer);
    return *this;
  }

  // Creates an APM instance with the specified config or the default one if
  // unspecified. Injects the specified components transferring the ownership
  // to the newly created APM instance.
  absl::Nullable<scoped_refptr<AudioProcessing>> Build(
      const Environment& env) override;

 private:
  AudioProcessing::Config config_;
  std::unique_ptr<EchoControlFactory> echo_control_factory_;
  std::unique_ptr<CustomProcessing> capture_post_processing_;
  std::unique_ptr<CustomProcessing> render_pre_processing_;
  scoped_refptr<EchoDetector> echo_detector_;
  std::unique_ptr<CustomAudioAnalyzer> capture_analyzer_;
};

}  // namespace webrtc

#endif  // API_AUDIO_BUILTIN_AUDIO_PROCESSING_BUILDER_H_
