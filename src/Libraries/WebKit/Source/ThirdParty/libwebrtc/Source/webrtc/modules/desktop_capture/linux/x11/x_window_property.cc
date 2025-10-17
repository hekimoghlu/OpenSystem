/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "modules/desktop_capture/linux/x11/x_window_property.h"

namespace webrtc {

XWindowPropertyBase::XWindowPropertyBase(Display* display,
                                         Window window,
                                         Atom property,
                                         int expected_size) {
  const int kBitsPerByte = 8;
  Atom actual_type;
  int actual_format;
  unsigned long bytes_after;  // NOLINT: type required by XGetWindowProperty
  int status = XGetWindowProperty(display, window, property, 0L, ~0L, False,
                                  AnyPropertyType, &actual_type, &actual_format,
                                  &size_, &bytes_after, &data_);
  if (status != Success) {
    data_ = nullptr;
    return;
  }
  if ((expected_size * kBitsPerByte) != actual_format) {
    size_ = 0;
    return;
  }

  is_valid_ = true;
}

XWindowPropertyBase::~XWindowPropertyBase() {
  if (data_)
    XFree(data_);
}

}  // namespace webrtc
