/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#ifndef MODULES_PORTAL_XDG_DESKTOP_PORTAL_UTILS_H_
#define MODULES_PORTAL_XDG_DESKTOP_PORTAL_UTILS_H_

#include <gio/gio.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "modules/portal/portal_request_response.h"
#include "modules/portal/scoped_glib.h"
#include "modules/portal/xdg_session_details.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace xdg_portal {

constexpr char kDesktopBusName[] = "org.freedesktop.portal.Desktop";
constexpr char kDesktopObjectPath[] = "/org/freedesktop/portal/desktop";
constexpr char kDesktopRequestObjectPath[] =
    "/org/freedesktop/portal/desktop/request";
constexpr char kSessionInterfaceName[] = "org.freedesktop.portal.Session";
constexpr char kRequestInterfaceName[] = "org.freedesktop.portal.Request";
constexpr char kScreenCastInterfaceName[] = "org.freedesktop.portal.ScreenCast";

using ProxyRequestCallback = void (*)(GObject*, GAsyncResult*, gpointer);
using SessionRequestCallback = void (*)(GDBusProxy*, GAsyncResult*, gpointer);
using SessionRequestResponseSignalHandler = void (*)(GDBusConnection*,
                                                     const char*,
                                                     const char*,
                                                     const char*,
                                                     const char*,
                                                     GVariant*,
                                                     gpointer);
using StartRequestResponseSignalHandler = void (*)(GDBusConnection*,
                                                   const char*,
                                                   const char*,
                                                   const char*,
                                                   const char*,
                                                   GVariant*,
                                                   gpointer);
using SessionStartRequestedHandler = void (*)(GDBusProxy*,
                                              GAsyncResult*,
                                              gpointer);

RTC_EXPORT std::string RequestResponseToString(RequestResponse request);

RequestResponse RequestResponseFromPortalResponse(uint32_t portal_response);

// Returns a string path for signal handle based on the provided connection and
// token.
RTC_EXPORT std::string PrepareSignalHandle(absl::string_view token,
                                           GDBusConnection* connection);

// Sets up the callback to execute when a response signal is received for the
// given object.
RTC_EXPORT uint32_t
SetupRequestResponseSignal(absl::string_view object_path,
                           const GDBusSignalCallback callback,
                           gpointer user_data,
                           GDBusConnection* connection);

RTC_EXPORT void RequestSessionProxy(
    absl::string_view interface_name,
    const ProxyRequestCallback proxy_request_callback,
    GCancellable* cancellable,
    gpointer user_data);

RTC_EXPORT void SetupSessionRequestHandlers(
    absl::string_view portal_prefix,
    const SessionRequestCallback session_request_callback,
    const SessionRequestResponseSignalHandler request_response_signale_handler,
    GDBusConnection* connection,
    GDBusProxy* proxy,
    GCancellable* cancellable,
    std::string& portal_handle,
    guint& session_request_signal_id,
    gpointer user_data);

RTC_EXPORT void StartSessionRequest(
    absl::string_view prefix,
    absl::string_view session_handle,
    const StartRequestResponseSignalHandler signal_handler,
    const SessionStartRequestedHandler session_started_handler,
    GDBusProxy* proxy,
    GDBusConnection* connection,
    GCancellable* cancellable,
    guint& start_request_signal_id,
    std::string& start_handle,
    gpointer user_data);

// Tears down the portal session and cleans up related objects.
RTC_EXPORT void TearDownSession(absl::string_view session_handle,
                                GDBusProxy* proxy,
                                GCancellable* cancellable,
                                GDBusConnection* connection);

}  // namespace xdg_portal
}  // namespace webrtc

#endif  // MODULES_PORTAL_XDG_DESKTOP_PORTAL_UTILS_H_
