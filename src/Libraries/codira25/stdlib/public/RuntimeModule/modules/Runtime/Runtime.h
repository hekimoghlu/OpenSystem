/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

//===--- Runtime.h - Codira runtime imports ----------------------*- C++ -*-===//
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
//
//  Things to drag in from the Codira runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BACKTRACING_RUNTIME_H
#define LANGUAGE_BACKTRACING_RUNTIME_H

#include <stdbool.h>
#include <stdlib.h>

#include "language/Runtime/CrashInfo.h"

#ifdef __cplusplus
extern "C" {
#endif

// Can't import language/Runtime/Debug.h because it assumes C++
void language_reportWarning(uint32_t flags, const char *message);

// Returns true if the given function is a thunk function
bool _language_backtrace_isThunkFunction(const char *rawName);

// Demangle the given raw name (supports Codira and C++)
char *_language_backtrace_demangle(const char *rawName,
                                size_t rawNameLength,
                                char *outputBuffer,
                                size_t *outputBufferSize);

#ifdef __cplusplus
}
#endif

#endif // LANGUAGE_BACKTRACING_RUNTIME_H
