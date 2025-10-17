/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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

//===- StringUtils.cpp - Routines for manipulating languagescan_string_ref_t -===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "language/DependencyScan/StringUtils.h"
#include <string>
#include <vector>

namespace language {
namespace c_string_utils {

languagescan_string_ref_t create_null() {
  languagescan_string_ref_t str;
  str.data = nullptr;
  str.length = 0;
  return str;
}

languagescan_string_ref_t create_clone(const char *string) {
  if (!string)
    return create_null();

  if (string[0] == '\0')
    return create_null();

  languagescan_string_ref_t str;
  str.data = strdup(string);
  str.length = strlen(string);
  return str;
}

languagescan_string_set_t *create_set(const std::vector<std::string> &strings) {
  languagescan_string_set_t *set = new languagescan_string_set_t;
  set->count = strings.size();
  set->strings = new languagescan_string_ref_t[set->count];
  for (unsigned SI = 0, SE = set->count; SI < SE; ++SI)
    set->strings[SI] = create_clone(strings[SI].c_str());
  return set;
}

languagescan_string_set_t *create_set(int count, const char **strings) {
  languagescan_string_set_t *set = new languagescan_string_set_t;
  set->count = count;
  set->strings = new languagescan_string_ref_t[set->count];
  for (unsigned SI = 0, SE = set->count; SI < SE; ++SI)
    set->strings[SI] = create_clone(strings[SI]);
  return set;
}

languagescan_string_set_t *create_empty_set() {
  languagescan_string_set_t *set = new languagescan_string_set_t;
  set->count = 0;
  return set;
}

const char *get_C_string(languagescan_string_ref_t string) {
  return static_cast<const char *>(string.data);
}
} // namespace c_string_utils
} // namespace language
