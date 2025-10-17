/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#ifndef MODULES_DESKTOP_CAPTURE_MOCK_DESKTOP_CAPTURER_CALLBACK_H_
#define MODULES_DESKTOP_CAPTURE_MOCK_DESKTOP_CAPTURER_CALLBACK_H_

#include <memory>

#include "modules/desktop_capture/desktop_capturer.h"
#include "test/gmock.h"

namespace webrtc {

class MockDesktopCapturerCallback : public DesktopCapturer::Callback {
 public:
  MockDesktopCapturerCallback();
  ~MockDesktopCapturerCallback() override;

  MockDesktopCapturerCallback(const MockDesktopCapturerCallback&) = delete;
  MockDesktopCapturerCallback& operator=(const MockDesktopCapturerCallback&) =
      delete;

  MOCK_METHOD(void,
              OnCaptureResultPtr,
              (DesktopCapturer::Result result,
               std::unique_ptr<DesktopFrame>* frame));
  void OnCaptureResult(DesktopCapturer::Result result,
                       std::unique_ptr<DesktopFrame> frame) final;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MOCK_DESKTOP_CAPTURER_CALLBACK_H_
