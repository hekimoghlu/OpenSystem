/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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

#include <stdio.h>
#include <stdlib.h>

int fib(int x) {
  if (x < 2) {
    if (x == 0) {
      return 0;
    }
    return 1;
  }

  return fib(x - 1) + fib(x - 2);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: fib <N> [<M>...]\n"
            "\n"
            "Return the Nth fibonacci number.\n");
    return 0;
  }

  for (int n = 1; n < argc; ++n) {
    int x = atoi(argv[n]);

    printf("%d: %d\n", x, fib(x));
  }

  return 0;
}
