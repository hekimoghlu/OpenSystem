/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#ifndef TEST_PC_E2E_ECHO_ECHO_EMULATION_H_
#define TEST_PC_E2E_ECHO_ECHO_EMULATION_H_

#include <atomic>
#include <deque>
#include <memory>
#include <vector>

#include "api/sequence_checker.h"
#include "api/test/pclf/media_configuration.h"
#include "modules/audio_device/include/test_audio_device.h"
#include "rtc_base/swap_queue.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Reduces audio input strength from provided capturer twice and adds input
// provided into EchoEmulatingCapturer::OnAudioRendered(...).
class EchoEmulatingCapturer : public TestAudioDeviceModule::Capturer {
 public:
  EchoEmulatingCapturer(
      std::unique_ptr<TestAudioDeviceModule::Capturer> capturer,
      EchoEmulationConfig config);

  void OnAudioRendered(rtc::ArrayView<const int16_t> data);

  int SamplingFrequency() const override {
    return delegate_->SamplingFrequency();
  }
  int NumChannels() const override { return delegate_->NumChannels(); }
  bool Capture(rtc::BufferT<int16_t>* buffer) override;

 private:
  std::unique_ptr<TestAudioDeviceModule::Capturer> delegate_;
  const EchoEmulationConfig config_;

  SwapQueue<std::vector<int16_t>> renderer_queue_;

  SequenceChecker renderer_thread_;
  std::vector<int16_t> queue_input_ RTC_GUARDED_BY(renderer_thread_);
  bool recording_started_ RTC_GUARDED_BY(renderer_thread_) = false;

  SequenceChecker capturer_thread_;
  std::vector<int16_t> queue_output_ RTC_GUARDED_BY(capturer_thread_);
  bool delay_accumulated_ RTC_GUARDED_BY(capturer_thread_) = false;
};

// Renders output into provided renderer and also copy output into provided
// EchoEmulationCapturer.
class EchoEmulatingRenderer : public TestAudioDeviceModule::Renderer {
 public:
  EchoEmulatingRenderer(
      std::unique_ptr<TestAudioDeviceModule::Renderer> renderer,
      EchoEmulatingCapturer* echo_emulating_capturer);

  int SamplingFrequency() const override {
    return delegate_->SamplingFrequency();
  }
  int NumChannels() const override { return delegate_->NumChannels(); }
  bool Render(rtc::ArrayView<const int16_t> data) override;

 private:
  std::unique_ptr<TestAudioDeviceModule::Renderer> delegate_;
  EchoEmulatingCapturer* echo_emulating_capturer_;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ECHO_ECHO_EMULATION_H_
