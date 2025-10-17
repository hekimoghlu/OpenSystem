/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "sysprop_helpers.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "sys/system_properties.h"

bool get_property_value(const char* property_name, char* dest, size_t dest_size) {
  assert(property_name && dest && dest_size != 0);
  const prop_info* prop = __system_property_find(property_name);
  if (!prop) return false;

  struct PropCbCookie {
    char* dest;
    size_t size;
  };
  *dest = '\0';
  PropCbCookie cb_cookie = {dest, dest_size};

  __system_property_read_callback(
      prop,
      [](void* cookie, const char* /* name */, const char* value, uint32_t /* serial */) {
        auto* cb_cookie = reinterpret_cast<PropCbCookie*>(cookie);
        strncpy(cb_cookie->dest, value, cb_cookie->size);
      },
      &cb_cookie);
  return *dest != '\0';
}

bool get_config_from_env_or_sysprops(const char* env_var_name, const char* const* sys_prop_names,
                                     size_t sys_prop_names_size, char* options,
                                     size_t options_size) {
  const char* env = getenv(env_var_name);
  if (env && *env != '\0') {
    strncpy(options, env, options_size);
    options[options_size - 1] = '\0';  // Ensure null-termination.
    return true;
  }

  for (size_t i = 0; i < sys_prop_names_size; ++i) {
    if (sys_prop_names[i] == nullptr) continue;
    if (get_property_value(sys_prop_names[i], options, options_size)) return true;
  }
  return false;
}
