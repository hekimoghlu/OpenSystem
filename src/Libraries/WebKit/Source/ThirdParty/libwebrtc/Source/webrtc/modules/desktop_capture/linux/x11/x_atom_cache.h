/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ATOM_CACHE_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ATOM_CACHE_H_

#include <X11/X.h>
#include <X11/Xlib.h>

namespace webrtc {

// A cache of Atom. Each Atom object is created on demand.
class XAtomCache final {
 public:
  explicit XAtomCache(::Display* display);
  ~XAtomCache();

  ::Display* display() const;

  Atom WmState();
  Atom WindowType();
  Atom WindowTypeNormal();
  Atom IccProfile();

 private:
  // If |*atom| is None, this function uses XInternAtom() to retrieve an Atom.
  Atom CreateIfNotExist(Atom* atom, const char* name);

  ::Display* const display_;
  Atom wm_state_ = None;
  Atom window_type_ = None;
  Atom window_type_normal_ = None;
  Atom icc_profile_ = None;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_X_ATOM_CACHE_H_
