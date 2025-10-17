/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

//===--- ErrorObjectTestSupport.h - Support for Instruments.app -*- C++ -*-===//
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
// Codira runtime support for tests involving errors.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_RUNTIME_ERROROBJECT_TEST_SUPPORT_H
#define LANGUAGE_RUNTIME_ERROROBJECT_TEST_SUPPORT_H

#include <atomic>

namespace language {

#if defined(__cplusplus)
LANGUAGE_RUNTIME_EXPORT std::atomic<void (*)(CodiraError *error)> _language_willThrow;
LANGUAGE_RUNTIME_EXPORT std::atomic<void (*)(
  OpaqueValue *value,
  const Metadata *type,
  const WitnessTable *errorConformance
)> _language_willThrowTypedImpl;
#endif

/// Set the value of @c _language_willThrow atomically.
///
/// This function is present for use by the standard library's test suite only.
LANGUAGE_CC(language)
LANGUAGE_RUNTIME_STDLIB_SPI
void _language_setWillThrowHandler(void (* handler)(CodiraError *error));
}

#endif
