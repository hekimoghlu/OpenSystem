/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
#include "Profiling.h"
#include <stdlib.h>

static unsigned *ArrayStart;
static unsigned NumElements;

/* EdgeProfAtExitHandler - When the program exits, just write out the profiling
 * data.
 */
static void EdgeProfAtExitHandler(void) {
  /* Note that if this were doing something more intelligent with the
   * instrumentation, we could do some computation here to expand what we
   * collected into simple edge profiles.  Since we directly count each edge, we
   * just write out all of the counters directly.
   */
  write_profiling_data(EdgeInfo, ArrayStart, NumElements);
}


/* llvm_start_edge_profiling - This is the main entry point of the edge
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_edge_profiling(int argc, const char **argv,
                              unsigned *arrayStart, unsigned numElements) {
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  NumElements = numElements;
  atexit(EdgeProfAtExitHandler);
  return Ret;
}
