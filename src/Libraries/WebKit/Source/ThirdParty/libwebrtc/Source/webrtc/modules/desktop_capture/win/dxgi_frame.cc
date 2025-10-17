/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#include "modules/desktop_capture/win/dxgi_frame.h"

#include <string.h>

#include <utility>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/win/dxgi_duplicator_controller.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

DxgiFrame::DxgiFrame(SharedMemoryFactory* factory) : factory_(factory) {}

DxgiFrame::~DxgiFrame() = default;

bool DxgiFrame::Prepare(DesktopSize size, DesktopCapturer::SourceId source_id) {
  if (source_id != source_id_) {
    // Once the source has been changed, the entire source should be copied.
    source_id_ = source_id;
    context_.Reset();
  }

  if (resolution_tracker_.SetResolution(size)) {
    // Once the output size changed, recreate the SharedDesktopFrame.
    frame_.reset();
  }

  if (!frame_) {
    std::unique_ptr<DesktopFrame> frame;
    if (factory_) {
      frame = SharedMemoryDesktopFrame::Create(size, factory_);

      if (!frame) {
        RTC_LOG(LS_WARNING) << "DxgiFrame cannot create a new DesktopFrame.";
        return false;
      }

      // DirectX capturer won't paint each pixel in the frame due to its one
      // capturer per monitor design. So once the new frame is created, we
      // should clear it to avoid the legacy image to be remained on it. See
      // http://crbug.com/708766.
      RTC_DCHECK_EQ(frame->stride(),
                    frame->size().width() * DesktopFrame::kBytesPerPixel);
      memset(frame->data(), 0, frame->stride() * frame->size().height());
    } else {
      frame.reset(new BasicDesktopFrame(size));
    }

    frame_ = SharedDesktopFrame::Wrap(std::move(frame));
  }

  return !!frame_;
}

SharedDesktopFrame* DxgiFrame::frame() const {
  RTC_DCHECK(frame_);
  return frame_.get();
}

DxgiFrame::Context* DxgiFrame::context() {
  RTC_DCHECK(frame_);
  return &context_;
}

}  // namespace webrtc
