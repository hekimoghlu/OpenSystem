/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "private/bionic_inline_raise.h"

void abort() {
  // Since abort() must not return, there's no error checking in this function:
  // there's no way to report an error anyway.

  // Unblock SIGABRT to give any signal handler a chance.
  sigset64_t mask;
  sigemptyset64(&mask);
  sigaddset64(&mask, SIGABRT);
  sigprocmask64(SIG_UNBLOCK, &mask, nullptr);

  // Use inline_raise() to raise SIGABRT without adding an uninteresting
  // stack frame that anyone investigating the crash would have to ignore.
  inline_raise(SIGABRT);

  // If that signal was ignored or was caught and the handler returned,
  // remove the signal handler and raise SIGABRT again.
  signal(SIGABRT, SIG_DFL);
  inline_raise(SIGABRT);

  // If we get this far, just exit.
  _exit(127);
}
