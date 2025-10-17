/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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

//===--- Exclusivity.cpp --------------------------------------------------===//
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
// This implements the runtime support for dynamically tracking exclusivity.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Exclusivity.h"
#include "language/Basic/Lazy.h"

#include <dlfcn.h>

void language::language_task_enterThreadLocalContextBackdeploy56(char *state) {
  const auto enterThreadLocalContext =
      reinterpret_cast<void(*)(char *state)>(LANGUAGE_LAZY_CONSTANT(
          dlsym(RTLD_DEFAULT, "language_task_enterThreadLocalContext")));
  if (enterThreadLocalContext)
    enterThreadLocalContext(state);
}

void language::language_task_exitThreadLocalContextBackdeploy56(char *state) {
  const auto exitThreadLocalContext =
      reinterpret_cast<void(*)(char *state)>(LANGUAGE_LAZY_CONSTANT(
          dlsym(RTLD_DEFAULT, "language_task_exitThreadLocalContext")));
  if (exitThreadLocalContext)
    exitThreadLocalContext(state);
}
