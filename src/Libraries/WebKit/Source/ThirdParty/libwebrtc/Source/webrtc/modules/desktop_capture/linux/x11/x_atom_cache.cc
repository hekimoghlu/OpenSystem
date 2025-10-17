/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#include "modules/desktop_capture/linux/x11/x_atom_cache.h"

#include "rtc_base/checks.h"

namespace webrtc {

XAtomCache::XAtomCache(::Display* display) : display_(display) {
  RTC_DCHECK(display_);
}

XAtomCache::~XAtomCache() = default;

::Display* XAtomCache::display() const {
  return display_;
}

Atom XAtomCache::WmState() {
  return CreateIfNotExist(&wm_state_, "WM_STATE");
}

Atom XAtomCache::WindowType() {
  return CreateIfNotExist(&window_type_, "_NET_WM_WINDOW_TYPE");
}

Atom XAtomCache::WindowTypeNormal() {
  return CreateIfNotExist(&window_type_normal_, "_NET_WM_WINDOW_TYPE_NORMAL");
}

Atom XAtomCache::IccProfile() {
  return CreateIfNotExist(&icc_profile_, "_ICC_PROFILE");
}

Atom XAtomCache::CreateIfNotExist(Atom* atom, const char* name) {
  RTC_DCHECK(atom);
  if (*atom == None) {
    *atom = XInternAtom(display(), name, True);
  }
  return *atom;
}

}  // namespace webrtc
