/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {

class WindowCapturerNull : public DesktopCapturer {
 public:
  WindowCapturerNull();
  ~WindowCapturerNull() override;

  WindowCapturerNull(const WindowCapturerNull&) = delete;
  WindowCapturerNull& operator=(const WindowCapturerNull&) = delete;

  // DesktopCapturer interface.
  void Start(Callback* callback) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* sources) override;
  bool SelectSource(SourceId id) override;

 private:
  Callback* callback_ = nullptr;
};

WindowCapturerNull::WindowCapturerNull() {}
WindowCapturerNull::~WindowCapturerNull() {}

bool WindowCapturerNull::GetSourceList(SourceList* sources) {
  // Not implemented yet.
  return false;
}

bool WindowCapturerNull::SelectSource(SourceId id) {
  // Not implemented yet.
  return false;
}

void WindowCapturerNull::Start(Callback* callback) {
  RTC_DCHECK(!callback_);
  RTC_DCHECK(callback);

  callback_ = callback;
}

void WindowCapturerNull::CaptureFrame() {
  // Not implemented yet.
  callback_->OnCaptureResult(Result::ERROR_TEMPORARY, nullptr);
}

}  // namespace

// static
std::unique_ptr<DesktopCapturer> DesktopCapturer::CreateRawWindowCapturer(
    const DesktopCaptureOptions& options) {
  return std::unique_ptr<DesktopCapturer>(new WindowCapturerNull());
}

}  // namespace webrtc
