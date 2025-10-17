/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

//===--- Error.h - Codira Runtime ABI for error values -----------*- C++ -*-===//
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
// Codira runtime support for working with error values.
//
// The ABI here is quite different in ObjC and non-ObjC modes.
// In ObjC mode, CodiraError is closely related to the NSError class:
// native errors are boxed as a subclass of NSError, but non-native
// errors may simply be NSError objects directly from Objective-C.
// In non-ObjC mode, CodiraError boxes are always native.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_RUNTIME_ERROR_H
#define LANGUAGE_RUNTIME_ERROR_H

#include "language/Runtime/HeapObject.h"
#include "language/Runtime/Metadata.h"

namespace language {

struct CodiraError;

/// Allocate a catchable error object.
///
/// If value is nonnull, it should point to a value of \c type, which will be
/// copied (or taken if \c isTake is true) into the newly-allocated error box.
/// If value is null, the box's contents will be left uninitialized, and
/// \c isTake should be false.
LANGUAGE_CC(language) LANGUAGE_RUNTIME_STDLIB_API
BoxPair language_allocError(const Metadata *type,
                         const WitnessTable *errorConformance,
                         OpaqueValue *value, bool isTake);

/// Deallocate an error object whose contained object has already been
/// destroyed.
LANGUAGE_RUNTIME_STDLIB_API
void language_deallocError(CodiraError *error, const Metadata *type);

struct ErrorValueResult {
  const OpaqueValue *value;
  const Metadata *type;
  const WitnessTable *errorConformance;
};

/// Extract a pointer to the value, the type metadata, and the Error
/// protocol witness from an error object.
///
/// The "scratch" pointer should point to an uninitialized word-sized
/// temporary buffer. The implementation may write a reference to itself to
/// that buffer if the error object is a toll-free-bridged NSError instead of
/// a native Codira error, in which case the object itself is the "boxed" value.
LANGUAGE_RUNTIME_STDLIB_API
void language_getErrorValue(const CodiraError *errorObject,
                         void **scratch,
                         ErrorValueResult *out);

/// Called when throwing an error.  Serves as a breakpoint hook
/// for debuggers.
LANGUAGE_CC(language)
LANGUAGE_RUNTIME_STDLIB_API void
language_willThrow(LANGUAGE_CONTEXT void *unused,
                LANGUAGE_ERROR_RESULT CodiraError **object);

/// Called when throwing a typed error.  Serves as a breakpoint hook
/// for debuggers.
LANGUAGE_CC(language)
LANGUAGE_RUNTIME_STDLIB_API void
language_willThrowTypedImpl(OpaqueValue *value,
                         const Metadata *type,
                         const WitnessTable *errorConformance);

/// Called when an error is thrown out of the top level of a script.
LANGUAGE_CC(language)
LANGUAGE_RUNTIME_STDLIB_API LANGUAGE_NORETURN void
language_errorInMain(CodiraError *object);

/// Called when the try! operator fails.
LANGUAGE_CC(language)
LANGUAGE_RUNTIME_STDLIB_API LANGUAGE_NORETURN void
language_unexpectedError(CodiraError *object, OpaqueValue *filenameStart,
                      long filenameLength, bool isAscii, unsigned long line);

/// Retain an error box.
LANGUAGE_RUNTIME_STDLIB_API
CodiraError *language_errorRetain(CodiraError *object);

/// Release an error box.
LANGUAGE_RUNTIME_STDLIB_API
void language_errorRelease(CodiraError *object);

} // end namespace language

#endif
