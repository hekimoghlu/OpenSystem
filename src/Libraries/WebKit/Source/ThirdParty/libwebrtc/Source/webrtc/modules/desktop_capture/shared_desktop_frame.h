/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_SHARED_DESKTOP_FRAME_H_
#define MODULES_DESKTOP_CAPTURE_SHARED_DESKTOP_FRAME_H_

#include <memory>

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_frame.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// SharedDesktopFrame is a DesktopFrame that may have multiple instances all
// sharing the same buffer.
class RTC_EXPORT SharedDesktopFrame final : public DesktopFrame {
 public:
  ~SharedDesktopFrame() override;

  SharedDesktopFrame(const SharedDesktopFrame&) = delete;
  SharedDesktopFrame& operator=(const SharedDesktopFrame&) = delete;

  static std::unique_ptr<SharedDesktopFrame> Wrap(
      std::unique_ptr<DesktopFrame> desktop_frame);

  // Deprecated.
  // TODO(sergeyu): remove this method.
  static SharedDesktopFrame* Wrap(DesktopFrame* desktop_frame);

  // Deprecated. Clients do not need to know the underlying DesktopFrame
  // instance.
  // TODO(zijiehe): Remove this method.
  // Returns the underlying instance of DesktopFrame.
  DesktopFrame* GetUnderlyingFrame();

  // Returns whether `this` and `other` share the underlying DesktopFrame.
  bool ShareFrameWith(const SharedDesktopFrame& other) const;

  // Creates a clone of this object.
  std::unique_ptr<SharedDesktopFrame> Share();

  // Checks if the frame is currently shared. If it returns false it's
  // guaranteed that there are no clones of the object.
  bool IsShared();

 private:
  typedef rtc::FinalRefCountedObject<std::unique_ptr<DesktopFrame>> Core;

  SharedDesktopFrame(rtc::scoped_refptr<Core> core);

  const rtc::scoped_refptr<Core> core_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SHARED_DESKTOP_FRAME_H_
