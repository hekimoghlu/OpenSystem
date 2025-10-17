/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

//===--- FoundationSupport.cpp - Support functions for Foundation ---------===//
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
// Helper functions for the Foundation framework.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_RUNTIME_FOUNDATION_SUPPORT_H
#define LANGUAGE_RUNTIME_FOUNDATION_SUPPORT_H

#include "language/Runtime/Config.h"

#if LANGUAGE_OBJC_INTEROP
#include <objc/runtime.h>

#ifdef __cplusplus
namespace language { extern "C" {
#endif

/// Returns a boolean indicating whether the Objective-C name of a class type is
/// stable across executions, i.e., if the class name is safe to serialize. (The
/// names of private and local types are unstable.)
LANGUAGE_RUNTIME_STDLIB_SPI
bool _language_isObjCTypeNameSerializable(Class theClass);

#ifdef __cplusplus
}} // extern "C", namespace language
#endif

#endif // LANGUAGE_OBJC_INTEROP
#endif // LANGUAGE_RUNTIME_FOUNDATION_SUPPORT_H
