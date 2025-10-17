/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

//===--- CommonString.h - C API for Codira Dependency Scanning ---*- C -*-===//
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

 #ifndef LANGUAGE_C_LIB_LANGUAGE_COMMON_STRING_H
 #define LANGUAGE_C_LIB_LANGUAGE_COMMON_STRING_H

 #include <stdbool.h>
 #include <stddef.h>
 #include <stdint.h>

 /**
  * A character string used to pass around dependency scan result metadata.
  * Lifetime of the string is strictly tied to the object whose field it
  * represents. When the owning object is released, string memory is freed.
  */
 typedef struct {
   const void *data;
   size_t length;
 } languagescan_string_ref_t;

 typedef struct {
   languagescan_string_ref_t *strings;
   size_t count;
 } languagescan_string_set_t;

 #endif // LANGUAGE_C_LIB_LANGUAGE_COMMON_STRING_H
