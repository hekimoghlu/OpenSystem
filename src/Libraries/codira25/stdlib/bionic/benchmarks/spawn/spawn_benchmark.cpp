/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "spawn_benchmark.h"

#include <errno.h>
#include <spawn.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <android-base/stringprintf.h>

extern char** environ;

void BM_spawn_test(benchmark::State& state, const char* const* argv) {
  for (auto _ : state) {
    pid_t child = 0;
    if (int spawn_err = posix_spawn(&child, argv[0], nullptr, nullptr, const_cast<char**>(argv),
                                    environ)) {
      state.SkipWithError(android::base::StringPrintf(
          "posix_spawn of %s failed: %s", argv[0], strerror(spawn_err)).c_str());
      break;
    }

    int wstatus = 0;
    const pid_t wait_result = TEMP_FAILURE_RETRY(waitpid(child, &wstatus, 0));
    if (wait_result != child) {
      state.SkipWithError(android::base::StringPrintf(
          "waitpid on pid %d for %s failed: %s",
          static_cast<int>(child), argv[0], strerror(errno)).c_str());
      break;
    }
    if (WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 127) {
      state.SkipWithError(android::base::StringPrintf("could not exec %s", argv[0]).c_str());
      break;
    }
  }
}
