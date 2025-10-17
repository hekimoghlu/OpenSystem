/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

//===--- CrashReporter.cpp - Crash Reporter integration ---------*- C++ -*-===//
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
// Declares gCRAnnotations.  This lets us link with other static libraries
// that also declare gCRAnnotations, because we'll pull in their copy
// (assuming they're linked first).
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Debug.h"

#if LANGUAGE_HAVE_CRASHREPORTERCLIENT

// Instead of linking to CrashReporterClient.a (because it complicates the
// build system), define the only symbol from that static archive ourselves.
//
// The layout of this struct is CrashReporter ABI, so there are no ABI concerns
// here.
extern "C" {
LANGUAGE_LIBRARY_VISIBILITY
struct crashreporter_annotations_t gCRAnnotations __attribute__((
    __section__("__DATA," CRASHREPORTER_ANNOTATIONS_SECTION))) = {
    CRASHREPORTER_ANNOTATIONS_VERSION, 0, 0, 0, 0, 0, 0, 0};
}

#endif
