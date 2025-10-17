/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "modules/portal/scoped_glib.h"

namespace webrtc {

template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<GError>;
template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<char>;
template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<GVariant>;
template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<GVariantIter>;
template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<GDBusMessage>;
template class RTC_EXPORT_TEMPLATE_DEFINE(RTC_EXPORT) Scoped<GUnixFDList>;

template <>
Scoped<GError>::~Scoped() {
  if (ptr_) {
    g_error_free(ptr_);
  }
}

template <>
Scoped<char>::~Scoped() {
  if (ptr_) {
    g_free(ptr_);
  }
}

template <>
Scoped<GVariant>::~Scoped() {
  if (ptr_) {
    g_variant_unref(ptr_);
  }
}

template <>
Scoped<GVariantIter>::~Scoped() {
  if (ptr_) {
    g_variant_iter_free(ptr_);
  }
}

template <>
Scoped<GDBusMessage>::~Scoped() {
  if (ptr_) {
    g_object_unref(ptr_);
  }
}

template <>
Scoped<GUnixFDList>::~Scoped() {
  if (ptr_) {
    g_object_unref(ptr_);
  }
}

}  // namespace webrtc
