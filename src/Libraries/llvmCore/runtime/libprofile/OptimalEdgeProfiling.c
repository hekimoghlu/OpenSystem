/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

/* OptEdgeProfAtExitHandler - When the program exits, just write out the
 * profiling data.
 */
static void OptEdgeProfAtExitHandler(void) {
  /* Note that, although the array has a counter for each edge, not all
   * counters are updated, the ones that are not used are initialised with -1.
   * When loading this information the counters with value -1 have to be
   * recalculated, it is guaranteed that this is possible.
   */
  write_profiling_data(OptEdgeInfo, ArrayStart, NumElements);
}


/* llvm_start_opt_edge_profiling - This is the main entry point of the edge
 * profiling library.  It is responsible for setting up the atexit handler.
 */
int llvm_start_opt_edge_profiling(int argc, const char **argv,
                                  unsigned *arrayStart, unsigned numElements) {
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  NumElements = numElements;
  atexit(OptEdgeProfAtExitHandler);
  return Ret;
}
