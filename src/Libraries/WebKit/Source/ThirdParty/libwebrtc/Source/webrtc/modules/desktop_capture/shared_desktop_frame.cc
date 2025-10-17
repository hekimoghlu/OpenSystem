/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include "modules/desktop_capture/shared_desktop_frame.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace webrtc {

SharedDesktopFrame::~SharedDesktopFrame() {}

// static
std::unique_ptr<SharedDesktopFrame> SharedDesktopFrame::Wrap(
    std::unique_ptr<DesktopFrame> desktop_frame) {
  return std::unique_ptr<SharedDesktopFrame>(new SharedDesktopFrame(
      rtc::scoped_refptr<Core>(new Core(std::move(desktop_frame)))));
}

SharedDesktopFrame* SharedDesktopFrame::Wrap(DesktopFrame* desktop_frame) {
  return Wrap(std::unique_ptr<DesktopFrame>(desktop_frame)).release();
}

DesktopFrame* SharedDesktopFrame::GetUnderlyingFrame() {
  return core_->get();
}

bool SharedDesktopFrame::ShareFrameWith(const SharedDesktopFrame& other) const {
  return core_->get() == other.core_->get();
}

std::unique_ptr<SharedDesktopFrame> SharedDesktopFrame::Share() {
  std::unique_ptr<SharedDesktopFrame> result(new SharedDesktopFrame(core_));
  result->CopyFrameInfoFrom(*this);
  return result;
}

bool SharedDesktopFrame::IsShared() {
  return !core_->HasOneRef();
}

SharedDesktopFrame::SharedDesktopFrame(rtc::scoped_refptr<Core> core)
    : DesktopFrame((*core)->size(),
                   (*core)->stride(),
                   (*core)->data(),
                   (*core)->shared_memory()),
      core_(core) {
  CopyFrameInfoFrom(*(core_->get()));
}

}  // namespace webrtc
