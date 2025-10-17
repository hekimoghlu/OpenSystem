/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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

#include <stdint.h>
#include <sys/cdefs.h>

// Get the presiding config string, in the following order of priority:
//   1. Environment variables.
//   2. System properties, in the order they're specified in sys_prop_names.
// If neither of these options are specified (or they're both an empty string),
// this function returns false. Otherwise, it returns true, and the presiding
// options string is written to the `options` buffer of size `size`. If this
// function returns true, `options` is guaranteed to be null-terminated.
// `options_size` should be at least PROP_VALUE_MAX.
__LIBC_HIDDEN__ bool get_config_from_env_or_sysprops(const char* env_var_name,
                                                     const char* const* sys_prop_names,
                                                     size_t sys_prop_names_size, char* options,
                                                     size_t options_size);

__LIBC_HIDDEN__ bool get_property_value(const char* property_name, char* dest, size_t dest_size);
