/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#include "modules/desktop_capture/cropping_window_capturer.h"

#include <stddef.h>

#include <utility>

#include "modules/desktop_capture/cropped_desktop_frame.h"
#include "rtc_base/logging.h"

namespace webrtc {

CroppingWindowCapturer::CroppingWindowCapturer(
    const DesktopCaptureOptions& options)
    : options_(options),
      callback_(NULL),
      window_capturer_(DesktopCapturer::CreateRawWindowCapturer(options)),
      selected_window_(kNullWindowId),
      excluded_window_(kNullWindowId) {}

CroppingWindowCapturer::~CroppingWindowCapturer() {}

void CroppingWindowCapturer::Start(DesktopCapturer::Callback* callback) {
  callback_ = callback;
  window_capturer_->Start(callback);
}

void CroppingWindowCapturer::SetSharedMemoryFactory(
    std::unique_ptr<SharedMemoryFactory> shared_memory_factory) {
  window_capturer_->SetSharedMemoryFactory(std::move(shared_memory_factory));
}

void CroppingWindowCapturer::CaptureFrame() {
  if (ShouldUseScreenCapturer()) {
    if (!screen_capturer_.get()) {
      screen_capturer_ = DesktopCapturer::CreateRawScreenCapturer(options_);
      if (excluded_window_) {
        screen_capturer_->SetExcludedWindow(excluded_window_);
      }
      screen_capturer_->Start(this);
    }
    screen_capturer_->CaptureFrame();
  } else {
    window_capturer_->CaptureFrame();
  }
}

void CroppingWindowCapturer::SetExcludedWindow(WindowId window) {
  excluded_window_ = window;
  if (screen_capturer_.get()) {
    screen_capturer_->SetExcludedWindow(window);
  }
}

bool CroppingWindowCapturer::GetSourceList(SourceList* sources) {
  return window_capturer_->GetSourceList(sources);
}

bool CroppingWindowCapturer::SelectSource(SourceId id) {
  if (window_capturer_->SelectSource(id)) {
    selected_window_ = id;
    return true;
  }
  return false;
}

bool CroppingWindowCapturer::FocusOnSelectedSource() {
  return window_capturer_->FocusOnSelectedSource();
}

void CroppingWindowCapturer::OnCaptureResult(
    DesktopCapturer::Result result,
    std::unique_ptr<DesktopFrame> screen_frame) {
  if (!ShouldUseScreenCapturer()) {
    RTC_LOG(LS_INFO) << "Window no longer on top when ScreenCapturer finishes";
    window_capturer_->CaptureFrame();
    return;
  }

  if (result != Result::SUCCESS) {
    RTC_LOG(LS_WARNING) << "ScreenCapturer failed to capture a frame";
    callback_->OnCaptureResult(result, nullptr);
    return;
  }

  DesktopRect window_rect = GetWindowRectInVirtualScreen();
  if (window_rect.is_empty()) {
    RTC_LOG(LS_WARNING) << "Window rect is empty";
    callback_->OnCaptureResult(Result::ERROR_TEMPORARY, nullptr);
    return;
  }

  std::unique_ptr<DesktopFrame> cropped_frame =
      CreateCroppedDesktopFrame(std::move(screen_frame), window_rect);

  if (!cropped_frame) {
    RTC_LOG(LS_WARNING) << "Window is outside of the captured display";
    callback_->OnCaptureResult(Result::ERROR_TEMPORARY, nullptr);
    return;
  }

  callback_->OnCaptureResult(Result::SUCCESS, std::move(cropped_frame));
}

bool CroppingWindowCapturer::IsOccluded(const DesktopVector& pos) {
  // Returns true if either capturer returns true.
  if (window_capturer_->IsOccluded(pos)) {
    return true;
  }
  if (screen_capturer_ != nullptr && screen_capturer_->IsOccluded(pos)) {
    return true;
  }
  return false;
}

#if !defined(WEBRTC_WIN)
// CroppingWindowCapturer is implemented only for windows. On other platforms
// the regular window capturer is used.
// static
std::unique_ptr<DesktopCapturer> CroppingWindowCapturer::CreateCapturer(
    const DesktopCaptureOptions& options) {
  return DesktopCapturer::CreateWindowCapturer(options);
}
#endif

}  // namespace webrtc
