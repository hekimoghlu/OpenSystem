/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

//===----------------------------------------------------------------------===//
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

#pragma once

#if __wasi__

#include <errno.h>
#include <fcntl.h>

// wasi-libc defines the following constants in a way that Clang Importer can't
// understand, so we need to expose them manually.
static inline int32_t _getConst_O_ACCMODE(void) { return O_ACCMODE; }
static inline int32_t _getConst_O_APPEND(void) { return O_APPEND; }
static inline int32_t _getConst_O_CREAT(void) { return O_CREAT; }
static inline int32_t _getConst_O_DIRECTORY(void) { return O_DIRECTORY; }
static inline int32_t _getConst_O_EXCL(void) { return O_EXCL; }
static inline int32_t _getConst_O_NONBLOCK(void) { return O_NONBLOCK; }
static inline int32_t _getConst_O_TRUNC(void) { return O_TRUNC; }
static inline int32_t _getConst_O_WRONLY(void) { return O_WRONLY; }

static inline int32_t _getConst_EWOULDBLOCK(void) { return EWOULDBLOCK; }
static inline int32_t _getConst_EOPNOTSUPP(void) { return EOPNOTSUPP; }

#endif
