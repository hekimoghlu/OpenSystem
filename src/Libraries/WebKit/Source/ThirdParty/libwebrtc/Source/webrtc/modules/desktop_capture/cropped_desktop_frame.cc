/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include "modules/desktop_capture/cropped_desktop_frame.h"

#include <memory>
#include <utility>

#include "modules/desktop_capture/desktop_region.h"
#include "rtc_base/checks.h"

namespace webrtc {

// A DesktopFrame that is a sub-rect of another DesktopFrame.
class CroppedDesktopFrame : public DesktopFrame {
 public:
  CroppedDesktopFrame(std::unique_ptr<DesktopFrame> frame,
                      const DesktopRect& rect);

  CroppedDesktopFrame(const CroppedDesktopFrame&) = delete;
  CroppedDesktopFrame& operator=(const CroppedDesktopFrame&) = delete;

 private:
  const std::unique_ptr<DesktopFrame> frame_;
};

std::unique_ptr<DesktopFrame> CreateCroppedDesktopFrame(
    std::unique_ptr<DesktopFrame> frame,
    const DesktopRect& rect) {
  RTC_DCHECK(frame);

  DesktopRect intersection = DesktopRect::MakeSize(frame->size());
  intersection.IntersectWith(rect);
  if (intersection.is_empty()) {
    return nullptr;
  }

  if (frame->size().equals(rect.size())) {
    return frame;
  }

  return std::unique_ptr<DesktopFrame>(
      new CroppedDesktopFrame(std::move(frame), intersection));
}

CroppedDesktopFrame::CroppedDesktopFrame(std::unique_ptr<DesktopFrame> frame,
                                         const DesktopRect& rect)
    : DesktopFrame(rect.size(),
                   frame->stride(),
                   frame->GetFrameDataAtPos(rect.top_left()),
                   frame->shared_memory()),
      frame_(std::move(frame)) {
  MoveFrameInfoFrom(frame_.get());
  set_top_left(frame_->top_left().add(rect.top_left()));
  mutable_updated_region()->IntersectWith(rect);
  mutable_updated_region()->Translate(-rect.left(), -rect.top());
}

}  // namespace webrtc
