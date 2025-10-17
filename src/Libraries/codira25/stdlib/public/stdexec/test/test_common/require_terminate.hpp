/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#pragma once

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#  include <unistd.h>
#  include <sys/wait.h>
#  define REQUIRE_TERMINATE __require_terminate
#endif

#include <cstdlib>
#include <exception>

#include <catch2/catch.hpp>

#ifdef REQUIRE_TERMINATE
namespace {
  template <class F, class... Args>
  void __require_terminate(F&& f, Args&&... args) {
    // spawn a new process
    auto child_pid = ::fork();

    // if the fork succeed
    if (child_pid >= 0) {

      // if we are in the child process
      if (child_pid == 0) {

        // call the function that we expect to abort
        std::set_terminate([] { std::exit(EXIT_FAILURE); });

        std::invoke(static_cast<F&&>(f), static_cast<Args&&>(args)...);

        // if the function didn't abort, we'll exit cleanly
        std::exit(EXIT_SUCCESS);
      }
    }

    // determine if the child process aborted
    int exit_status{};
    ::wait(&exit_status);

    // we check the exit status instead of a signal interrupt, because
    // Catch is going to catch the signal and exit with an error
    bool aborted = WEXITSTATUS(exit_status);
    if (!aborted) {
      INFO("He didn't fall? Inconceivable!");
    }
    REQUIRE(aborted);
  }
} // namespace
#endif
