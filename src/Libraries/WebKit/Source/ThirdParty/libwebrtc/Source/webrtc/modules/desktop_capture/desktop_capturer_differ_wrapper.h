/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURER_DIFFER_WRAPPER_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURER_DIFFER_WRAPPER_H_

#include <memory>
#if defined(WEBRTC_USE_GIO)
#include "modules/desktop_capture/desktop_capture_metadata.h"
#endif  // defined(WEBRTC_USE_GIO)
#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "modules/desktop_capture/shared_memory.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// DesktopCapturer wrapper that calculates updated_region() by comparing frames
// content. This class always expects the underlying DesktopCapturer
// implementation returns a superset of updated regions in DestkopFrame. If a
// DesktopCapturer implementation does not know the updated region, it should
// set updated_region() to full frame.
//
// This class marks entire frame as updated if the frame size or frame stride
// has been changed.
class RTC_EXPORT DesktopCapturerDifferWrapper
    : public DesktopCapturer,
      public DesktopCapturer::Callback {
 public:
  // Creates a DesktopCapturerDifferWrapper with a DesktopCapturer
  // implementation, and takes its ownership.
  explicit DesktopCapturerDifferWrapper(
      std::unique_ptr<DesktopCapturer> base_capturer);

  ~DesktopCapturerDifferWrapper() override;

  // DesktopCapturer interface.
  void Start(DesktopCapturer::Callback* callback) override;
  void SetSharedMemoryFactory(
      std::unique_ptr<SharedMemoryFactory> shared_memory_factory) override;
  void CaptureFrame() override;
  void SetExcludedWindow(WindowId window) override;
  bool GetSourceList(SourceList* screens) override;
  bool SelectSource(SourceId id) override;
  bool FocusOnSelectedSource() override;
  bool IsOccluded(const DesktopVector& pos) override;
#if defined(WEBRTC_USE_GIO)
  DesktopCaptureMetadata GetMetadata() override;
#endif  // defined(WEBRTC_USE_GIO)
 private:
  // DesktopCapturer::Callback interface.
  void OnCaptureResult(Result result,
                       std::unique_ptr<DesktopFrame> frame) override;

  const std::unique_ptr<DesktopCapturer> base_capturer_;
  DesktopCapturer::Callback* callback_;
  std::unique_ptr<SharedDesktopFrame> last_frame_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURER_DIFFER_WRAPPER_H_
