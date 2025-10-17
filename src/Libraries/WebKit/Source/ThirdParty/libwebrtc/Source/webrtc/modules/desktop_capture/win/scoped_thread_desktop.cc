/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include "modules/desktop_capture/win/scoped_thread_desktop.h"

#include "modules/desktop_capture/win/desktop.h"

namespace webrtc {

ScopedThreadDesktop::ScopedThreadDesktop()
    : initial_(Desktop::GetThreadDesktop()) {}

ScopedThreadDesktop::~ScopedThreadDesktop() {
  Revert();
}

bool ScopedThreadDesktop::IsSame(const Desktop& desktop) {
  if (assigned_.get() != NULL) {
    return assigned_->IsSame(desktop);
  } else {
    return initial_->IsSame(desktop);
  }
}

void ScopedThreadDesktop::Revert() {
  if (assigned_.get() != NULL) {
    initial_->SetThreadDesktop();
    assigned_.reset();
  }
}

bool ScopedThreadDesktop::SetThreadDesktop(Desktop* desktop) {
  Revert();

  std::unique_ptr<Desktop> scoped_desktop(desktop);

  if (initial_->IsSame(*desktop))
    return true;

  if (!desktop->SetThreadDesktop())
    return false;

  assigned_.reset(scoped_desktop.release());
  return true;
}

}  // namespace webrtc
