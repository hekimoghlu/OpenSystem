/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

//===----- StringUtils.h - Managed C String Utility Functions -----*- C -*-===//
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

#include "language-c/CommonString/CommonString.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/StringSet.h"
#include <string>
#include <vector>

//=== Private Utility Functions--------------------------------------------===//
namespace language {
namespace c_string_utils {

/// Create null string
languagescan_string_ref_t create_null();

/// Create a \c languagescan_string_ref_t object from a nul-terminated C string.  New
/// \c languagescan_string_ref_t will contain a copy of \p string.
languagescan_string_ref_t create_clone(const char *string);

/// Create an array of \c languagescan_string_ref_t objects from a vector of C++ strings using the
/// create_clone routine.
languagescan_string_set_t *create_set(const std::vector<std::string> &strings);

/// Create an array of languagescan_string_ref_t objects from an array of C strings using the
/// create_clone routine.
languagescan_string_set_t *create_set(int count, const char **strings);

/// Create an empty array of languagescan_string_ref_t objects
languagescan_string_set_t *create_empty_set();

/// Retrieve the character data associated with the given string.
const char *get_C_string(languagescan_string_ref_t string);
}
}
