/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include <stdarg.h>
#include <sys/ptrace.h>

extern "C" long __ptrace(int req, pid_t pid, void* addr, void* data);

long ptrace(int req, ...) {
  bool is_peek = (req == PTRACE_PEEKUSR || req == PTRACE_PEEKTEXT || req == PTRACE_PEEKDATA);
  long peek_result;

  va_list args;
  va_start(args, req);
  pid_t pid = va_arg(args, pid_t);
  void* addr = va_arg(args, void*);
  void* data;
  if (is_peek) {
    data = &peek_result;
  } else {
    data = va_arg(args, void*);
  }
  va_end(args);

  long result = __ptrace(req, pid, addr, data);
  if (is_peek && result == 0) {
    return peek_result;
  }
  return result;
}
