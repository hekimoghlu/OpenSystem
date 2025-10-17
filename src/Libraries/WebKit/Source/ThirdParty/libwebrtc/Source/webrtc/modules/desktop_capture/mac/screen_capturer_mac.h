/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_SCREEN_CAPTURER_MAC_H_
#define MODULES_DESKTOP_CAPTURE_MAC_SCREEN_CAPTURER_MAC_H_

#include <CoreGraphics/CoreGraphics.h>

#include <memory>
#include <vector>

#include "api/sequence_checker.h"
#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/desktop_region.h"
#include "modules/desktop_capture/mac/desktop_configuration.h"
#include "modules/desktop_capture/mac/desktop_configuration_monitor.h"
#include "modules/desktop_capture/mac/desktop_frame_provider.h"
#include "modules/desktop_capture/screen_capture_frame_queue.h"
#include "modules/desktop_capture/screen_capturer_helper.h"
#include "modules/desktop_capture/shared_desktop_frame.h"

namespace webrtc {

class DisplayStreamManager;

// A class to perform video frame capturing for mac.
class ScreenCapturerMac final : public DesktopCapturer {
 public:
  ScreenCapturerMac(rtc::scoped_refptr<DesktopConfigurationMonitor> desktop_config_monitor,
                    bool detect_updated_region,
                    bool allow_iosurface);
  ~ScreenCapturerMac() override;

  ScreenCapturerMac(const ScreenCapturerMac&) = delete;
  ScreenCapturerMac& operator=(const ScreenCapturerMac&) = delete;

  // TODO(julien.isorce): Remove Init() or make it private.
  bool Init();

  // DesktopCapturer interface.
  void Start(Callback* callback) override;
  void CaptureFrame() override;
  void SetExcludedWindow(WindowId window) override;
  bool GetSourceList(SourceList* screens) override;
  bool SelectSource(SourceId id) override;

 private:
  // Returns false if the selected screen is no longer valid.
  bool CgBlit(const DesktopFrame& frame, const DesktopRegion& region);

  // Called when the screen configuration is changed.
  void ScreenConfigurationChanged();

  bool RegisterRefreshAndMoveHandlers();
  void UnregisterRefreshAndMoveHandlers();

  void ScreenRefresh(CGDirectDisplayID display_id,
                     CGRectCount count,
                     const CGRect* rect_array,
                     DesktopVector display_origin,
                     IOSurfaceRef io_surface);
  void ReleaseBuffers();

  std::unique_ptr<DesktopFrame> CreateFrame();

  const bool detect_updated_region_;

  Callback* callback_ = nullptr;

  // Queue of the frames buffers.
  ScreenCaptureFrameQueue<SharedDesktopFrame> queue_;

  // Current display configuration.
  MacDesktopConfiguration desktop_config_;

  // Currently selected display, or 0 if the full desktop is selected. On OS X
  // 10.6 and before, this is always 0.
  CGDirectDisplayID current_display_ = 0;

  // The physical pixel bounds of the current screen.
  DesktopRect screen_pixel_bounds_;

  // The dip to physical pixel scale of the current screen.
  float dip_to_pixel_scale_ = 1.0f;

  // A thread-safe list of invalid rectangles, and the size of the most
  // recently captured screen.
  ScreenCapturerHelper helper_;

  // Contains an invalid region from the previous capture.
  DesktopRegion last_invalid_region_;

  // Monitoring display reconfiguration.
  rtc::scoped_refptr<DesktopConfigurationMonitor> desktop_config_monitor_;

  CGWindowID excluded_window_ = 0;

  // List of streams, one per screen.
  std::vector<CGDisplayStreamRef> display_streams_;

  // Container holding latest state of the snapshot per displays.
  DesktopFrameProvider desktop_frame_provider_;

  // Start, CaptureFrame and destructor have to called in the same thread.
  SequenceChecker thread_checker_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_SCREEN_CAPTURER_MAC_H_
