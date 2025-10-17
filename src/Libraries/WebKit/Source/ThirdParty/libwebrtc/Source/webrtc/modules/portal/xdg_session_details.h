/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#ifndef MODULES_PORTAL_XDG_SESSION_DETAILS_H_
#define MODULES_PORTAL_XDG_SESSION_DETAILS_H_

#include <gio/gio.h>

#include <string>

namespace webrtc {
namespace xdg_portal {

// Details of the session associated with XDG desktop portal session. Portal API
// calls can be invoked by utilizing the information here.
struct SessionDetails {
  GDBusProxy* proxy = nullptr;
  GCancellable* cancellable = nullptr;
  std::string session_handle;
  uint32_t pipewire_stream_node_id = 0;
};

}  // namespace xdg_portal
}  // namespace webrtc

#endif  // MODULES_PORTAL_XDG_SESSION_DETAILS_H_
