/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#include <string.h>

#include "bionic/pthread_internal.h"

// Maps regular signals like SIGSEGV to strings like "Segmentation fault".
// Signal 0 and all the real-time signals are just nullptr, but that's the ABI.
const char* const sys_siglist[NSIG] = {
#define __BIONIC_SIGDEF(signal_number, signal_description) [signal_number] = signal_description,
#include "private/bionic_sigdefs.h"
};

// Maps regular signals like SIGSEGV to strings like "SEGV".
// Signal 0 and all the real-time signals are just nullptr, but that's the ABI.
const char* const sys_signame[NSIG] = {
#define __BIONIC_SIGDEF(signal_number, unused) [signal_number] = &(#signal_number)[3],
#include "private/bionic_sigdefs.h"
};

extern "C" __LIBC_HIDDEN__ const char* __strsignal(int signal_number, char* buf, size_t buf_len) {
  if (signal_number >= SIGHUP && signal_number < SIGSYS) {
    return sys_siglist[signal_number];
  }
  const char* prefix = "Unknown";
  if (signal_number >= SIGRTMIN && signal_number <= SIGRTMAX) {
    prefix = "Real-time";
    signal_number -= SIGRTMIN;
  }
  size_t length = snprintf(buf, buf_len, "%s signal %d", prefix, signal_number);
  if (length >= buf_len) {
    return nullptr;
  }
  return buf;
}

char* strsignal(int signal_number) {
  bionic_tls& tls = __get_bionic_tls();
  return const_cast<char*>(__strsignal(signal_number, tls.strsignal_buf, sizeof(tls.strsignal_buf)));
}

int sig2str(int sig, char* str) {
  if (sig >= SIGHUP && sig <= SIGSYS) {
    strcpy(str, sys_signame[sig]);
    return 0;
  }
  if (sig == SIGRTMIN) {
    strcpy(str, "RTMIN");
    return 0;
  }
  if (sig == SIGRTMAX) {
    strcpy(str, "RTMAX");
    return 0;
  }
  if (sig > SIGRTMIN && sig < SIGRTMAX) {
    if (sig - SIGRTMIN <= SIGRTMAX - sig) {
      sprintf(str, "RTMIN+%d", sig - SIGRTMIN);
    } else {
      sprintf(str, "RTMAX-%d", SIGRTMAX - sig);
    }
    return 0;
  }
  return -1;
}

int str2sig(const char* str, int* sig) {
  // A name in our list, like "SEGV"?
  for (size_t i = SIGHUP; i <= SIGSYS; ++i) {
    if (!strcmp(str, sys_signame[i])) {
      *sig = i;
      return 0;
    }
  }

  // The two named special cases?
  if (!strcmp(str, "RTMIN")) {
    *sig = SIGRTMIN;
    return 0;
  }
  if (!strcmp(str, "RTMAX")) {
    *sig = SIGRTMAX;
    return 0;
  }

  // Must be either an integer corresponding to a regular signal such as "9",
  // or a string of the form "RTMIN+%d" or "RTMAX-%d".
  int base = 0;
  if (!strncmp(str, "RTMIN+", 6)) {
    base = SIGRTMIN;
    str += 5;
  } else if (!strncmp(str, "RTMAX-", 6)) {
    base = SIGRTMAX;
    str += 5;
  }
  char* end = nullptr;
  errno = 0;
  int offset = strtol(str, &end, 10);
  if (errno || *end) return -1;

  // Reject out of range integers (like "666"),
  // and out of range real-time signals (like "RTMIN+666" or "RTMAX-666").
  int result = base + offset;
  bool regular = (base == 0 && result >= SIGHUP && result <= SIGSYS);
  bool realtime = (result >= SIGRTMIN && result <= SIGRTMAX);
  if (!regular && !realtime) return -1;

  *sig = result;
  return 0;
}
