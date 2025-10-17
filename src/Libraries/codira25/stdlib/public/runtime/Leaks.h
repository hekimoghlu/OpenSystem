/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

//===--- Leaks.h ------------------------------------------------*- C++ -*-===//
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
// This is a very simple leak detector implementation that detects objects that
// are allocated but not deallocated in a region. It is purposefully behind a
// flag since it is not meant to be used in production yet.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_STDLIB_RUNTIME_LEAKS_H
#define LANGUAGE_STDLIB_RUNTIME_LEAKS_H

#if LANGUAGE_RUNTIME_ENABLE_LEAK_CHECKER

#include "language/shims/Visibility.h"

#include "language/Runtime/Config.h"

namespace language {
struct HeapObject;
}

LANGUAGE_CC(language)
LANGUAGE_RUNTIME_EXPORT LANGUAGE_NOINLINE LANGUAGE_USED void
_language_leaks_startTrackingObjects(const char *);

LANGUAGE_CC(language)
LANGUAGE_RUNTIME_EXPORT LANGUAGE_NOINLINE LANGUAGE_USED int
_language_leaks_stopTrackingObjects(const char *);

LANGUAGE_RUNTIME_EXPORT LANGUAGE_NOINLINE LANGUAGE_USED void
_language_leaks_startTrackingObject(language::HeapObject *);

LANGUAGE_RUNTIME_EXPORT LANGUAGE_NOINLINE LANGUAGE_USED void
_language_leaks_stopTrackingObject(language::HeapObject *);

#define LANGUAGE_LEAKS_START_TRACKING_OBJECT(obj)                                 \
  _language_leaks_startTrackingObject(obj)
#define LANGUAGE_LEAKS_STOP_TRACKING_OBJECT(obj)                                  \
  _language_leaks_stopTrackingObject(obj)

// LANGUAGE_RUNTIME_ENABLE_LEAK_CHECKER
#else
// not LANGUAGE_RUNTIME_ENABLE_LEAK_CHECKER

#define LANGUAGE_LEAKS_START_TRACKING_OBJECT(obj)
#define LANGUAGE_LEAKS_STOP_TRACKING_OBJECT(obj)

#endif

#endif
