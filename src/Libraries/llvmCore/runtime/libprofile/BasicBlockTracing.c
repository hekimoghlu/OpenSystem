/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#include <stdio.h>

static unsigned *ArrayStart, *ArrayEnd, *ArrayCursor;

/* WriteAndFlushBBTraceData - write out the currently accumulated trace data
 * and reset the cursor to point to the beginning of the buffer.
 */
static void WriteAndFlushBBTraceData () {
  write_profiling_data(BBTraceInfo, ArrayStart, (ArrayCursor - ArrayStart));
  ArrayCursor = ArrayStart;
}

/* BBTraceAtExitHandler - When the program exits, just write out any remaining 
 * data and free the trace buffer.
 */
static void BBTraceAtExitHandler(void) {
  WriteAndFlushBBTraceData ();
  free (ArrayStart);
}

/* llvm_trace_basic_block - called upon hitting a new basic block. */
void llvm_trace_basic_block (unsigned BBNum) {
  *ArrayCursor++ = BBNum;
  if (ArrayCursor == ArrayEnd)
    WriteAndFlushBBTraceData ();
}

/* llvm_start_basic_block_tracing - This is the main entry point of the basic
 * block tracing library.  It is responsible for setting up the atexit
 * handler and allocating the trace buffer.
 */
int llvm_start_basic_block_tracing(int argc, const char **argv,
                              unsigned *arrayStart, unsigned numElements) {
  int Ret;
  const unsigned BufferSize = 128 * 1024;
  unsigned ArraySize;

  Ret = save_arguments(argc, argv);

  /* Allocate a buffer to contain BB tracing data */
  ArraySize = BufferSize / sizeof (unsigned);
  ArrayStart = malloc (ArraySize * sizeof (unsigned));
  ArrayEnd = ArrayStart + ArraySize;
  ArrayCursor = ArrayStart;

  /* Set up the atexit handler. */
  atexit (BBTraceAtExitHandler);

  return Ret;
}
