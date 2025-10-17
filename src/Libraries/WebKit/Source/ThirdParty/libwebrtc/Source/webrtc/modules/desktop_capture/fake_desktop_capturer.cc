/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#include "modules/desktop_capture/fake_desktop_capturer.h"

#include <utility>

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_frame.h"

namespace webrtc {

FakeDesktopCapturer::FakeDesktopCapturer() = default;
FakeDesktopCapturer::~FakeDesktopCapturer() = default;

void FakeDesktopCapturer::set_result(DesktopCapturer::Result result) {
  result_ = result;
}

int FakeDesktopCapturer::num_frames_captured() const {
  return num_frames_captured_;
}

int FakeDesktopCapturer::num_capture_attempts() const {
  return num_capture_attempts_;
}

// Uses the `generator` provided as DesktopFrameGenerator, FakeDesktopCapturer
// does
// not take the ownership of `generator`.
void FakeDesktopCapturer::set_frame_generator(
    DesktopFrameGenerator* generator) {
  generator_ = generator;
}

void FakeDesktopCapturer::Start(DesktopCapturer::Callback* callback) {
  callback_ = callback;
}

void FakeDesktopCapturer::CaptureFrame() {
  num_capture_attempts_++;
  if (generator_) {
    if (result_ != DesktopCapturer::Result::SUCCESS) {
      callback_->OnCaptureResult(result_, nullptr);
      return;
    }

    std::unique_ptr<DesktopFrame> frame(
        generator_->GetNextFrame(shared_memory_factory_.get()));
    if (frame) {
      num_frames_captured_++;
      callback_->OnCaptureResult(result_, std::move(frame));
    } else {
      callback_->OnCaptureResult(DesktopCapturer::Result::ERROR_TEMPORARY,
                                 nullptr);
    }
    return;
  }
  callback_->OnCaptureResult(DesktopCapturer::Result::ERROR_PERMANENT, nullptr);
}

void FakeDesktopCapturer::SetSharedMemoryFactory(
    std::unique_ptr<SharedMemoryFactory> shared_memory_factory) {
  shared_memory_factory_ = std::move(shared_memory_factory);
}

bool FakeDesktopCapturer::GetSourceList(DesktopCapturer::SourceList* sources) {
  sources->push_back({kWindowId, "A-Fake-DesktopCapturer-Window"});
  sources->push_back({kScreenId});
  return true;
}

bool FakeDesktopCapturer::SelectSource(DesktopCapturer::SourceId id) {
  return id == kWindowId || id == kScreenId || id == kFullDesktopScreenId;
}

}  // namespace webrtc
