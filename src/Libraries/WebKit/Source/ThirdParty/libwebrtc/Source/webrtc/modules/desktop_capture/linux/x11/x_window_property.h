/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_X_WINDOW_PROPERTY_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_X_WINDOW_PROPERTY_H_

#include <X11/X.h>
#include <X11/Xlib.h>

namespace webrtc {

class XWindowPropertyBase {
 public:
  XWindowPropertyBase(Display* display,
                      Window window,
                      Atom property,
                      int expected_size);
  virtual ~XWindowPropertyBase();

  XWindowPropertyBase(const XWindowPropertyBase&) = delete;
  XWindowPropertyBase& operator=(const XWindowPropertyBase&) = delete;

  // True if we got properly value successfully.
  bool is_valid() const { return is_valid_; }

  // Size and value of the property.
  size_t size() const { return size_; }

 protected:
  unsigned char* data_ = nullptr;

 private:
  bool is_valid_ = false;
  unsigned long size_ = 0;  // NOLINT: type required by XGetWindowProperty
};

// Convenience wrapper for XGetWindowProperty() results.
template <class PropertyType>
class XWindowProperty : public XWindowPropertyBase {
 public:
  XWindowProperty(Display* display, const Window window, const Atom property)
      : XWindowPropertyBase(display, window, property, sizeof(PropertyType)) {}
  ~XWindowProperty() override = default;

  XWindowProperty(const XWindowProperty&) = delete;
  XWindowProperty& operator=(const XWindowProperty&) = delete;

  const PropertyType* data() const {
    return reinterpret_cast<PropertyType*>(data_);
  }
  PropertyType* data() { return reinterpret_cast<PropertyType*>(data_); }
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_X_WINDOW_PROPERTY_H_
