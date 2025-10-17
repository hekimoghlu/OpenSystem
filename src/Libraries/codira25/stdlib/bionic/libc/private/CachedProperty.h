/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#pragma once

#include <string.h>
#include <sys/system_properties.h>

// Cached system property lookup. For code that needs to read the same property multiple times,
// this class helps optimize those lookups.
class CachedProperty {
 public:
  // The lifetime of `property_name` must be greater than that of this CachedProperty.
  explicit CachedProperty(const char* property_name)
    : property_name_(property_name),
      prop_info_(nullptr),
      cached_area_serial_(0),
      cached_property_serial_(0),
      is_read_only_(strncmp(property_name, "ro.", 3) == 0),
      read_only_property_(nullptr) {
    cached_value_[0] = '\0';
  }

  // Returns true if the property has been updated (based on the serial rather than the value)
  // since the last call to Get.
  bool DidChange() {
    uint32_t initial_property_serial_ = cached_property_serial_;
    Get();
    return (cached_property_serial_ != initial_property_serial_);
  }

  // Returns the current value of the underlying system property as cheaply as possible.
  // The returned pointer is valid until the next call to Get. It is the caller's responsibility
  // to provide a lock for thread-safety.
  const char* Get() {
    // Do we have a `struct prop_info` yet?
    if (prop_info_ == nullptr) {
      // `__system_property_find` is expensive, so only retry if a property
      // has been created since last time we checked.
      uint32_t property_area_serial = __system_property_area_serial();
      if (property_area_serial != cached_area_serial_) {
        prop_info_ = __system_property_find(property_name_);
        cached_area_serial_ = property_area_serial;
      }
    }

    if (prop_info_ != nullptr) {
      // Only bother re-reading the property if it's actually changed since last time.
      uint32_t property_serial = __system_property_serial(prop_info_);
      if (property_serial != cached_property_serial_) {
        __system_property_read_callback(prop_info_, &CachedProperty::Callback, this);
      }
    }
    if (is_read_only_ && read_only_property_ != nullptr) {
      return read_only_property_;
    }
    return cached_value_;
  }

 private:
  const char* property_name_;
  const prop_info* prop_info_;
  uint32_t cached_area_serial_;
  uint32_t cached_property_serial_;
  char cached_value_[PROP_VALUE_MAX];
  bool is_read_only_;
  const char* read_only_property_;

  static void Callback(void* data, const char*, const char* value, uint32_t serial) {
    CachedProperty* instance = reinterpret_cast<CachedProperty*>(data);
    instance->cached_property_serial_ = serial;
    // Read only properties can be larger than PROP_VALUE_MAX, but also never change value or
    // location, thus we return the pointer from the shared memory directly.
    if (instance->is_read_only_) {
      instance->read_only_property_ = value;
    } else {
      strlcpy(instance->cached_value_, value, PROP_VALUE_MAX);
    }
  }
};
