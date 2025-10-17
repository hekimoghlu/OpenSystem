/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

//===--- InterceptTraps.cpp -----------------------------------------------===//
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

#include <stdio.h>
#include <signal.h>
#include <string.h>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(__wasi__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include "language/Runtime/Config.h"

static void CrashCatcher(int Sig) {
  const char *Msg;
  switch (Sig) {
    case SIGILL:  Msg = "CRASHED: SIGILL\n";  break;
    case SIGABRT: Msg = "CRASHED: SIGABRT\n"; break;
    case SIGFPE:  Msg = "CRASHED: SIGFPE\n";  break;
    case SIGSEGV: Msg = "CRASHED: SIGSEGV\n"; break;
#if !defined(_WIN32)
    case SIGTRAP: Msg = "CRASHED: SIGTRAP\n"; break;
    case SIGBUS:  Msg = "CRASHED: SIGBUS\n";  break;
    case SIGSYS:  Msg = "CRASHED: SIGSYS\n";  break;
#endif
    default:      Msg = "CRASHED: SIG????\n"; break;
  }
#if defined(_WIN32)
  _write(_fileno(stderr), Msg, strlen(Msg));
#else
  write(STDERR_FILENO, Msg, strlen(Msg));
#endif
  _exit(0);
}

#if defined(_WIN32)
static LONG WINAPI
VectoredCrashHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_ILLEGAL_INSTRUCTION:
    _write(_fileno(stderr), "CRASHED: SIGTRAP\n", 17);
    _exit(0);

  case EXCEPTION_DATATYPE_MISALIGNMENT:
    _write(_fileno(stderr), "CRASHED: SIGBUS\n", 16);
    _exit(0);
  }

  return EXCEPTION_CONTINUE_SEARCH;
}
#endif // _WIN32

LANGUAGE_CC(language) LANGUAGE_RUNTIME_LIBRARY_VISIBILITY extern "C"
void installTrapInterceptor() {
  // Disable buffering on stdout so that everything is printed before crashing.
  setbuf(stdout, 0);

#if defined(_WIN32)
  _set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif

  signal(SIGILL,  CrashCatcher);
  signal(SIGABRT, CrashCatcher);
  signal(SIGFPE,  CrashCatcher);
  signal(SIGSEGV, CrashCatcher);
#if defined(_WIN32)
  AddVectoredExceptionHandler(TRUE, VectoredCrashHandler);
#else
  signal(SIGTRAP, CrashCatcher);
  signal(SIGBUS,  CrashCatcher);
  signal(SIGSYS,  CrashCatcher);
#endif
}
