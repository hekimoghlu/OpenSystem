/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include <sys/wait.h>
#include <stddef.h>

extern "C" int __waitid(idtype_t which, id_t id, siginfo_t* info, int options, struct rusage* ru);

pid_t wait(int* status) {
  return wait4(-1, status, 0, nullptr);
}

pid_t waitpid(pid_t pid, int* status, int options) {
  return wait4(pid, status, options, nullptr);
}

int waitid(idtype_t which, id_t id, siginfo_t* info, int options) {
  // The system call takes an optional struct rusage that we don't need.
  return __waitid(which, id, info, options, nullptr);
}
