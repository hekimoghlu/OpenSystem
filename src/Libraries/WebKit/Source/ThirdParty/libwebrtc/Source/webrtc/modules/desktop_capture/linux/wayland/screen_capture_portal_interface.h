/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREEN_CAPTURE_PORTAL_INTERFACE_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREEN_CAPTURE_PORTAL_INTERFACE_H_

#include <gio/gio.h>

#include <string>

#include "modules/portal/portal_request_response.h"
#include "modules/portal/scoped_glib.h"
#include "modules/portal/xdg_desktop_portal_utils.h"
#include "modules/portal/xdg_session_details.h"

namespace webrtc {
namespace xdg_portal {

using SessionClosedSignalHandler = void (*)(GDBusConnection*,
                                            const char*,
                                            const char*,
                                            const char*,
                                            const char*,
                                            GVariant*,
                                            gpointer);

// A base class for XDG desktop portals that can capture desktop/screen.
// Note: downstream clients inherit from this class so it is advisable to
// provide a default implementation of any new virtual methods that may be added
// to this class.
class RTC_EXPORT ScreenCapturePortalInterface {
 public:
  virtual ~ScreenCapturePortalInterface() {}
  // Gets details about the session such as session handle.
  virtual xdg_portal::SessionDetails GetSessionDetails() { return {}; }
  // Starts the portal setup.
  virtual void Start() {}

  // Stops and cleans up the portal.
  virtual void Stop() {}

  // Notifies observers about the success/fail state of the portal
  // request/response.
  virtual void OnPortalDone(xdg_portal::RequestResponse result) {}
  // Sends a create session request to the portal.
  virtual void RequestSession(GDBusProxy* proxy) {}

  // Following methods should not be made virtual as they share a common
  // implementation between portals.

  // Requests portal session using the proxy object.
  void RequestSessionUsingProxy(GAsyncResult* result);
  // Handles the session request result.
  void OnSessionRequestResult(GDBusProxy* proxy, GAsyncResult* result);
  // Subscribes to session close signal and sets up a handler for it.
  void RegisterSessionClosedSignalHandler(
      const SessionClosedSignalHandler session_close_signal_handler,
      GVariant* parameters,
      GDBusConnection* connection,
      std::string& session_handle,
      guint& session_closed_signal_id);
  // Handles the result of session start request.
  void OnStartRequestResult(GDBusProxy* proxy, GAsyncResult* result);
};

}  // namespace xdg_portal
}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_SCREEN_CAPTURE_PORTAL_INTERFACE_H_
