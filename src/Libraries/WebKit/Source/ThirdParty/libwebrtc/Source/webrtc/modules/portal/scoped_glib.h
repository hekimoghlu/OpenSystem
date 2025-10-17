/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#ifndef MODULES_PORTAL_SCOPED_GLIB_H_
#define MODULES_PORTAL_SCOPED_GLIB_H_

#include <gio/gio.h>

#include "rtc_base/checks.h"
#include "rtc_base/system/rtc_export_template.h"

namespace webrtc {

template <class T>
class Scoped {
 public:
  Scoped() {}
  explicit Scoped(T* val) { ptr_ = val; }
  ~Scoped() { RTC_DCHECK_NOTREACHED(); }

  T* operator->() const { return ptr_; }

  explicit operator bool() const { return ptr_ != nullptr; }

  bool operator!() const { return ptr_ == nullptr; }

  T* get() const { return ptr_; }

  T** receive() {
    RTC_CHECK(!ptr_);
    return &ptr_;
  }

  Scoped& operator=(T* val) {
    RTC_DCHECK(val);
    ptr_ = val;
    return *this;
  }

 protected:
  T* ptr_ = nullptr;
};

template <>
Scoped<GError>::~Scoped();
template <>
Scoped<char>::~Scoped();
template <>
Scoped<GVariant>::~Scoped();
template <>
Scoped<GVariantIter>::~Scoped();
template <>
Scoped<GDBusMessage>::~Scoped();
template <>
Scoped<GUnixFDList>::~Scoped();

extern template class RTC_EXPORT_TEMPLATE_DECLARE(RTC_EXPORT) Scoped<GError>;
extern template class RTC_EXPORT_TEMPLATE_DECLARE(RTC_EXPORT) Scoped<char>;
extern template class RTC_EXPORT_TEMPLATE_DECLARE(RTC_EXPORT) Scoped<GVariant>;
extern template class RTC_EXPORT_TEMPLATE_DECLARE(
    RTC_EXPORT) Scoped<GVariantIter>;
extern template class RTC_EXPORT_TEMPLATE_DECLARE(
    RTC_EXPORT) Scoped<GDBusMessage>;
extern template class RTC_EXPORT_TEMPLATE_DECLARE(
    RTC_EXPORT) Scoped<GUnixFDList>;

}  // namespace webrtc

#endif  // MODULES_PORTAL_SCOPED_GLIB_H_
