/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "tool_setup.h"
#include "tool_stderr.h"
#include "tool_msgs.h"

#include "memdebug.h" /* keep this as LAST include */

FILE *tool_stderr;

void tool_init_stderr(void)
{
  /* !checksrc! disable STDERR 1 */
  tool_stderr = stderr;
}

void tool_set_stderr_file(struct GlobalConfig *global, char *filename)
{
  FILE *fp;

  if(!filename)
    return;

  if(!strcmp(filename, "-")) {
    tool_stderr = stdout;
    return;
  }

  /* precheck that filename is accessible to lessen the chance that the
     subsequent freopen will fail. */
  fp = fopen(filename, FOPEN_WRITETEXT);
  if(!fp) {
    warnf(global, "Warning: Failed to open %s", filename);
    return;
  }
  fclose(fp);

  /* freopen the actual stderr (stdio.h stderr) instead of tool_stderr since
     the latter may be set to stdout. */
  /* !checksrc! disable STDERR 1 */
  fp = freopen(filename, FOPEN_WRITETEXT, stderr);
  if(!fp) {
    /* stderr may have been closed by freopen. there is nothing to be done. */
    DEBUGASSERT(0);
    return;
  }
  /* !checksrc! disable STDERR 1 */
  tool_stderr = stderr;
}
