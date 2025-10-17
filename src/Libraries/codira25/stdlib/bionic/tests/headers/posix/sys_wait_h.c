/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <sys/wait.h>

#include "header_checks.h"

static void sys_wait_h() {
  MACRO(WCONTINUED);
  MACRO(WNOHANG);
  MACRO(WUNTRACED);

#if !defined(WEXITSTATUS)
#error WEXITSTATUS
#endif
#if !defined(WIFCONTINUED)
#error WIFCONTINUED
#endif
#if !defined(WIFEXITED)
#error WIFEXITED
#endif
#if !defined(WIFSIGNALED)
#error WIFSIGNALED
#endif
#if !defined(WIFSTOPPED)
#error WIFSTOPPED
#endif
#if !defined(WSTOPSIG)
#error WSTOPSIG
#endif
#if !defined(WTERMSIG)
#error WTERMSIG
#endif

  MACRO(WEXITED);
  MACRO(WNOWAIT);
  MACRO(WSTOPPED);

  TYPE(idtype_t);
  MACRO(P_ALL);
  MACRO(P_PGID);
  MACRO(P_PID);

  TYPE(id_t);
  TYPE(pid_t);
  TYPE(siginfo_t);
  TYPE(union sigval);

  FUNCTION(wait, pid_t (*f)(int*));
  FUNCTION(waitid, int (*f)(idtype_t, id_t, siginfo_t*, int));
  FUNCTION(waitpid, pid_t (*f)(pid_t, int*, int));
}
