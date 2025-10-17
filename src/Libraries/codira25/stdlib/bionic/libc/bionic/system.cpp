/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include <errno.h>
#include <stdlib.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "private/__bionic_get_shell_path.h"
#include "private/ScopedSignalBlocker.h"
#include "private/ScopedSignalHandler.h"

int system(const char* command) {
  // "The system() function shall always return non-zero when command is NULL."
  // https://pubs.opengroup.org/onlinepubs/9799919799.2024edition/functions/system.html
  if (command == nullptr) return 1;

  ScopedSignalBlocker sigchld_blocker(SIGCHLD);
  ScopedSignalHandler sigint_ignorer(SIGINT, SIG_IGN);
  ScopedSignalHandler sigquit_ignorer(SIGQUIT, SIG_IGN);

  sigset64_t default_mask = {};
  if (sigint_ignorer.old_action_.sa_handler != SIG_IGN) sigaddset64(&default_mask, SIGINT);
  if (sigquit_ignorer.old_action_.sa_handler != SIG_IGN) sigaddset64(&default_mask, SIGQUIT);

  static constexpr int flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;
  posix_spawnattr_t attributes;
  if ((errno = posix_spawnattr_init(&attributes))) return -1;
  if ((errno = posix_spawnattr_setsigdefault64(&attributes, &default_mask))) return -1;
  if ((errno = posix_spawnattr_setsigmask64(&attributes, &sigchld_blocker.old_set_))) return -1;
  if ((errno = posix_spawnattr_setflags(&attributes, flags))) return -1;

  const char* argv[] = {"sh", "-c", "--", command, nullptr};
  pid_t child;
  if ((errno = posix_spawn(&child, __bionic_get_shell_path(), nullptr, &attributes,
                           const_cast<char**>(argv), environ)) != 0) {
    return -1;
  }

  posix_spawnattr_destroy(&attributes);

  int status;
  pid_t pid = TEMP_FAILURE_RETRY(waitpid(child, &status, 0));
  return (pid == -1 ? -1 : status);
}
