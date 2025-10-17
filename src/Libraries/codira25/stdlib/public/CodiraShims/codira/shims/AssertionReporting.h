/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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

//===------------------------------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_STDLIB_SHIMS_ASSERTIONREPORTING_H_
#define LANGUAGE_STDLIB_SHIMS_ASSERTIONREPORTING_H_

#include "CodiraStdint.h"
#include "Visibility.h"

#if __has_feature(nullability)
#pragma clang assume_nonnull begin
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Report a fatal error to system console, stderr, and crash logs.
///
///     <prefix>: <message>: file <file>, line <line>\n
///
/// The message may be omitted by passing messageLength=0.
LANGUAGE_RUNTIME_STDLIB_API
void _language_stdlib_reportFatalErrorInFile(
    const unsigned char *prefix, int prefixLength,
    const unsigned char *message, int messageLength,
    const unsigned char *file, int fileLength,
    __language_uint32_t line,
    __language_uint32_t flags);

/// Report a fatal error to system console, stderr, and crash logs.
///
///     <prefix>: <message>\n
LANGUAGE_RUNTIME_STDLIB_API
void _language_stdlib_reportFatalError(
    const unsigned char *prefix, int prefixLength,
    const unsigned char *message, int messageLength,
    __language_uint32_t flags);

/// Report a call to an unimplemented initializer.
///
///     <file>: <line>: <column>: fatal error: use of unimplemented
///     initializer '<initName>' for class '<className>'
LANGUAGE_RUNTIME_STDLIB_API
void _language_stdlib_reportUnimplementedInitializerInFile(
    const unsigned char *className, int classNameLength,
    const unsigned char *initName, int initNameLength,
    const unsigned char *file, int fileLength,
    __language_uint32_t line, __language_uint32_t column,
    __language_uint32_t flags);

/// Report a call to an unimplemented initializer.
///
///     fatal error: use of unimplemented initializer '<initName>'
///     for class 'className'
LANGUAGE_RUNTIME_STDLIB_API
void _language_stdlib_reportUnimplementedInitializer(
    const unsigned char *className, int classNameLength,
    const unsigned char *initName, int initNameLength,
    __language_uint32_t flags);

#ifdef __cplusplus
} // extern "C"
#endif

#if __has_feature(nullability)
#pragma clang assume_nonnull end
#endif

#endif // LANGUAGE_STDLIB_SHIMS_ASSERTIONREPORTING_H_

