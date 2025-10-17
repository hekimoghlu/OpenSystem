/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SCOPED_GDI_HANDLE_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SCOPED_GDI_HANDLE_H_

#include <windows.h>

namespace webrtc {
namespace win {

// Scoper for GDI objects.
template <class T, class Traits>
class ScopedGDIObject {
 public:
  ScopedGDIObject() : handle_(NULL) {}
  explicit ScopedGDIObject(T object) : handle_(object) {}

  ~ScopedGDIObject() { Traits::Close(handle_); }

  ScopedGDIObject(const ScopedGDIObject&) = delete;
  ScopedGDIObject& operator=(const ScopedGDIObject&) = delete;

  T Get() { return handle_; }

  void Set(T object) {
    if (handle_ && object != handle_)
      Traits::Close(handle_);
    handle_ = object;
  }

  ScopedGDIObject& operator=(T object) {
    Set(object);
    return *this;
  }

  T release() {
    T object = handle_;
    handle_ = NULL;
    return object;
  }

  operator T() { return handle_; }

 private:
  T handle_;
};

// The traits class that uses DeleteObject() to close a handle.
template <typename T>
class DeleteObjectTraits {
 public:
  DeleteObjectTraits() = delete;
  DeleteObjectTraits(const DeleteObjectTraits&) = delete;
  DeleteObjectTraits& operator=(const DeleteObjectTraits&) = delete;

  // Closes the handle.
  static void Close(T handle) {
    if (handle)
      DeleteObject(handle);
  }
};

// The traits class that uses DestroyCursor() to close a handle.
class DestroyCursorTraits {
 public:
  DestroyCursorTraits() = delete;
  DestroyCursorTraits(const DestroyCursorTraits&) = delete;
  DestroyCursorTraits& operator=(const DestroyCursorTraits&) = delete;

  // Closes the handle.
  static void Close(HCURSOR handle) {
    if (handle)
      DestroyCursor(handle);
  }
};

typedef ScopedGDIObject<HBITMAP, DeleteObjectTraits<HBITMAP> > ScopedBitmap;
typedef ScopedGDIObject<HCURSOR, DestroyCursorTraits> ScopedCursor;

}  // namespace win
}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SCOPED_GDI_HANDLE_H_
