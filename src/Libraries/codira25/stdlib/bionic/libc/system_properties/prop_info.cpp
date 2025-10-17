/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include "system_properties/prop_info.h"

#include <string.h>

static constexpr const char kLongLegacyError[] =
    "Must use __system_property_read_callback() to read";
static_assert(sizeof(kLongLegacyError) < prop_info::kLongLegacyErrorBufferSize,
              "Error message for long properties read by legacy libc must fit within 56 chars");

prop_info::prop_info(const char* name, uint32_t namelen, const char* value, uint32_t valuelen) {
  memcpy(this->name, name, namelen);
  this->name[namelen] = '\0';
  atomic_store_explicit(&this->serial, valuelen << 24, memory_order_relaxed);
  memcpy(this->value, value, valuelen);
  this->value[valuelen] = '\0';
}

prop_info::prop_info(const char* name, uint32_t namelen, uint32_t long_offset) {
  memcpy(this->name, name, namelen);
  this->name[namelen] = '\0';

  auto error_value_len = sizeof(kLongLegacyError) - 1;
  atomic_store_explicit(&this->serial, error_value_len << 24 | kLongFlag, memory_order_relaxed);
  memcpy(this->long_property.error_message, kLongLegacyError, sizeof(kLongLegacyError));

  this->long_property.offset = long_offset;
}
