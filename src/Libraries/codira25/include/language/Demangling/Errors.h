/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

//===--- Errors.h - Demangling library error handling -----------*- C++ -*-===//
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
// This file exists because not every client links to liblanguageCore (the
// runtime), so calling language::fatalError() or language::warning() from within
// the demangler is not an option.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DEMANGLING_ERRORS_H
#define LANGUAGE_DEMANGLING_ERRORS_H

#include "language/Demangling/NamespaceMacros.h"
#include <inttypes.h>
#include <stdarg.h>

#ifndef LANGUAGE_FORMAT
// LANGUAGE_FORMAT(fmt,first) marks a function as taking a format string argument
// at argument `fmt`, with the first argument for the format string as `first`.
#if __has_attribute(format)
#define LANGUAGE_FORMAT(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#define LANGUAGE_FORMAT(fmt, first)
#endif
#endif

#ifndef LANGUAGE_VFORMAT
// LANGUAGE_VFORMAT(fmt) marks a function as taking a format string argument at
// argument `fmt`, with the arguments in a `va_list`.
#if __has_attribute(format)
#define LANGUAGE_VFORMAT(fmt) __attribute__((format(printf, fmt, 0)))
#else
#define LANGUAGE_VFORMAT(fmt)
#endif
#endif

#ifndef LANGUAGE_NORETURN
#if __has_attribute(noreturn)
#define LANGUAGE_NORETURN __attribute__((__noreturn__))
#else
#define LANGUAGE_NORETURN
#endif
#endif

namespace language {
namespace Demangle {
LANGUAGE_BEGIN_INLINE_NAMESPACE

LANGUAGE_NORETURN LANGUAGE_FORMAT(2, 3) void fatal(uint32_t flags, const char *format,
                                             ...);
LANGUAGE_FORMAT(2, 3) void warn(uint32_t flags, const char *format, ...);

LANGUAGE_NORETURN LANGUAGE_VFORMAT(2) void fatalv(uint32_t flags, const char *format,
                                            va_list val);
LANGUAGE_VFORMAT(2) void warnv(uint32_t flags, const char *format, va_list val);

LANGUAGE_END_INLINE_NAMESPACE
} // end namespace Demangle
} // end namespace language

#endif // LANGUAGE_DEMANGLING_DEMANGLE_H
