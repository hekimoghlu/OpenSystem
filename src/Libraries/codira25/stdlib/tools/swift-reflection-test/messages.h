/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

//===--- messages.h - Remote reflection testing messages ------------------===//
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

static const char *REQUEST_INSTANCE_KIND = "k\n";
static const char *REQUEST_SHOULD_UNWRAP_CLASS_EXISTENTIAL = "u\n";
static const char *REQUEST_INSTANCE_ADDRESS = "i\n";
static const char *REQUEST_REFLECTION_INFO = "r\n";
static const char *REQUEST_IMAGES = "m\n";
static const char *REQUEST_READ_BYTES = "b\n";
static const char *REQUEST_SYMBOL_ADDRESS = "s\n";
static const char *REQUEST_STRING_LENGTH = "l\n";
static const char *REQUEST_POINTER_SIZE = "p\n";
static const char *REQUEST_DONE = "d\n";

typedef enum InstanceKind {
  None,
  Object,
  Existential,
  ErrorExistential,
  Closure,
  Enum,
  EnumValue,
  AsyncTask,
  LogString,
} InstanceKind;
