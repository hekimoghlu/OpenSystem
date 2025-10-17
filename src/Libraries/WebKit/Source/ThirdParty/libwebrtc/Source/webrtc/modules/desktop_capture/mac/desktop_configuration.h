/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_H_

#include <ApplicationServices/ApplicationServices.h>

#include <vector>

#include "modules/desktop_capture/desktop_geometry.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Describes the configuration of a specific display.
struct MacDisplayConfiguration {
  MacDisplayConfiguration();
  MacDisplayConfiguration(const MacDisplayConfiguration& other);
  MacDisplayConfiguration(MacDisplayConfiguration&& other);
  ~MacDisplayConfiguration();

  MacDisplayConfiguration& operator=(const MacDisplayConfiguration& other);
  MacDisplayConfiguration& operator=(MacDisplayConfiguration&& other);

  // Cocoa identifier for this display.
  CGDirectDisplayID id = 0;

  // Bounds of this display in Density-Independent Pixels (DIPs).
  DesktopRect bounds;

  // Bounds of this display in physical pixels.
  DesktopRect pixel_bounds;

  // Scale factor from DIPs to physical pixels.
  float dip_to_pixel_scale = 1.0f;

  // Display type, built-in or external.
  bool is_builtin;
};

typedef std::vector<MacDisplayConfiguration> MacDisplayConfigurations;

// Describes the configuration of the whole desktop.
struct RTC_EXPORT MacDesktopConfiguration {
  // Used to request bottom-up or top-down coordinates.
  enum Origin { BottomLeftOrigin, TopLeftOrigin };

  MacDesktopConfiguration();
  MacDesktopConfiguration(const MacDesktopConfiguration& other);
  MacDesktopConfiguration(MacDesktopConfiguration&& other);
  ~MacDesktopConfiguration();

  MacDesktopConfiguration& operator=(const MacDesktopConfiguration& other);
  MacDesktopConfiguration& operator=(MacDesktopConfiguration&& other);

  // Returns the desktop & display configurations.
  // If BottomLeftOrigin is used, the output is in Cocoa-style "bottom-up"
  // (the origin is the bottom-left of the primary monitor, and coordinates
  // increase as you move up the screen). Otherwise, the configuration will be
  // converted to follow top-left coordinate system as Windows and X11.
  static MacDesktopConfiguration GetCurrent(Origin origin);

  // Returns true if the given desktop configuration equals this one.
  bool Equals(const MacDesktopConfiguration& other);

  // If `id` corresponds to the built-in display, return its configuration,
  // otherwise return the configuration for the display with the specified id,
  // or nullptr if no such display exists.
  const MacDisplayConfiguration* FindDisplayConfigurationById(
      CGDirectDisplayID id);

  // Bounds of the desktop excluding monitors with DPI settings different from
  // the main monitor. In Density-Independent Pixels (DIPs).
  DesktopRect bounds;

  // Same as bounds, but expressed in physical pixels.
  DesktopRect pixel_bounds;

  // Scale factor from DIPs to physical pixels.
  float dip_to_pixel_scale = 1.0f;

  // Configurations of the displays making up the desktop area.
  MacDisplayConfigurations displays;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_H_
