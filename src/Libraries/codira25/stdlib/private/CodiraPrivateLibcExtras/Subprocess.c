/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
//===--- Subprocess.c - Subprocess Stubs ----------------------------------===//
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

// posix_spawn is not available on Android, HAIKU, WASI or Windows (MSVC).
#if !defined(__ANDROID__) && !defined(__HAIKU__) && (!defined(_WIN32) || defined(__CYGWIN__)) && !defined(__wasi__)

#include "language/Runtime/Config.h"

// NOTE: preprocess away the availability information to allow use of
// unsupported APIs on certain targets (i.e. tvOS)
#define availability(...)
#include <spawn.h>
#include <sys/types.h>

LANGUAGE_CC(language)
int _stdlib_posix_spawn_file_actions_init(
    posix_spawn_file_actions_t *file_actions) {
  return posix_spawn_file_actions_init(file_actions);
}

LANGUAGE_CC(language)
int _stdlib_posix_spawn_file_actions_destroy(
    posix_spawn_file_actions_t *file_actions) {
  return posix_spawn_file_actions_destroy(file_actions);
}

LANGUAGE_CC(language)
int _stdlib_posix_spawn_file_actions_addclose(
    posix_spawn_file_actions_t *file_actions, int filedes) {
  return posix_spawn_file_actions_addclose(file_actions, filedes);
}

LANGUAGE_CC(language)
int _stdlib_posix_spawn_file_actions_adddup2(
    posix_spawn_file_actions_t *file_actions, int filedes, int newfiledes) {
  return posix_spawn_file_actions_adddup2(file_actions, filedes, newfiledes);
}

LANGUAGE_CC(language)
int _stdlib_posix_spawn(pid_t *__restrict pid, const char * __restrict path,
                      const posix_spawn_file_actions_t *file_actions,
                      const posix_spawnattr_t *__restrict attrp,
                      char *const argv[__restrict],
                      char *const envp[__restrict]) {
  return posix_spawn(pid, path, file_actions, attrp, argv, envp);
}

#endif // !defined(__ANDROID__) && !defined(__HAIKU__) && (!defined(_WIN32) || defined(__CYGWIN__)) && !defined(__wasi__)

