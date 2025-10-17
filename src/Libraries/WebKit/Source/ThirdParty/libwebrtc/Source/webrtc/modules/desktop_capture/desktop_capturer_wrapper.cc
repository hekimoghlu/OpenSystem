/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include "modules/desktop_capture/desktop_capturer_wrapper.h"

#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

DesktopCapturerWrapper::DesktopCapturerWrapper(
    std::unique_ptr<DesktopCapturer> base_capturer)
    : base_capturer_(std::move(base_capturer)) {
  RTC_DCHECK(base_capturer_);
}

DesktopCapturerWrapper::~DesktopCapturerWrapper() = default;

void DesktopCapturerWrapper::Start(Callback* callback) {
  base_capturer_->Start(callback);
}

void DesktopCapturerWrapper::SetSharedMemoryFactory(
    std::unique_ptr<SharedMemoryFactory> shared_memory_factory) {
  base_capturer_->SetSharedMemoryFactory(std::move(shared_memory_factory));
}

void DesktopCapturerWrapper::CaptureFrame() {
  base_capturer_->CaptureFrame();
}

void DesktopCapturerWrapper::SetExcludedWindow(WindowId window) {
  base_capturer_->SetExcludedWindow(window);
}

bool DesktopCapturerWrapper::GetSourceList(SourceList* sources) {
  return base_capturer_->GetSourceList(sources);
}

bool DesktopCapturerWrapper::SelectSource(SourceId id) {
  return base_capturer_->SelectSource(id);
}

bool DesktopCapturerWrapper::FocusOnSelectedSource() {
  return base_capturer_->FocusOnSelectedSource();
}

bool DesktopCapturerWrapper::IsOccluded(const DesktopVector& pos) {
  return base_capturer_->IsOccluded(pos);
}

}  // namespace webrtc
