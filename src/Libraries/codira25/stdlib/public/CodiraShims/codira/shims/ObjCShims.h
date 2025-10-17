/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

//===--- ObjCShims.h - Access to libobjc for the core stdlib ---------===//
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
//  Using the ObjectiveC module in the core stdlib would create a
//  circular dependency, so instead we import these declarations as
//  part of CodiraShims.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_STDLIB_SHIMS_OBJCSHIMS_H
#define LANGUAGE_STDLIB_SHIMS_OBJCSHIMS_H

#include "CodiraStdint.h"
#include "Visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __OBJC2__

extern void objc_setAssociatedObject(void);
extern void objc_getAssociatedObject(void);
int objc_sync_enter(id _Nonnull object);
int objc_sync_exit(id _Nonnull object);

static void * _Nonnull getSetAssociatedObjectPtr() {
  return (void *)&objc_setAssociatedObject;
}

static void * _Nonnull getGetAssociatedObjectPtr() {
  return (void *)&objc_getAssociatedObject;
}

#endif // __OBJC2__

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LANGUAGE_STDLIB_SHIMS_COREFOUNDATIONSHIMS_H

