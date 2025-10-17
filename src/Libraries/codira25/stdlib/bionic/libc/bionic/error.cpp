/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include <error.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned int error_message_count = 0;
void (*error_print_progname)(void) = nullptr;
int error_one_per_line = 0;

static void __error_head() {
  ++error_message_count;

  if (error_print_progname != nullptr) {
    error_print_progname();
  } else {
    fflush(stdout);
    fprintf(stderr, "%s:", getprogname());
  }
}

static void __error_tail(int status, int error) {
  if (error != 0) {
    fprintf(stderr, ": %s", strerror(error));
  }

  putc('\n', stderr);
  fflush(stderr);

  if (status != 0) {
    exit(status);
  }
}

void error(int status, int error, const char* fmt, ...) {
  __error_head();
  putc(' ', stderr);

  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  __error_tail(status, error);
}

void error_at_line(int status, int error, const char* file, unsigned int line, const char* fmt, ...) {
  if (error_one_per_line) {
    static const char* last_file;
    static unsigned int last_line;
    if (last_line == line && last_file != nullptr && strcmp(last_file, file) == 0) {
      return;
    }
    last_file = file;
    last_line = line;
  }

  __error_head();
  fprintf(stderr, "%s:%d: ", file, line);

  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  __error_tail(status, error);
}
