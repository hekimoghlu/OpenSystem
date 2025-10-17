/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#include "libqhull.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/*-<a                             href="qh-user.htm#TOC"
   >-------------------------------</a><a name="qh_fprintf_rbox">-</a>

   qh_fprintf_rbox(fp, msgcode, format, list of args )
     print arguments to *fp according to format
     Use qh_fprintf_rbox() for rboxlib.c

   notes:
     same as fprintf()
     fgets() is not trapped like fprintf()
     exit qh_fprintf_rbox via qh_errexit_rbox()
*/

void qh_fprintf_rbox(FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;

    if (!fp) {
        fprintf(stderr, "QH6231 Qhull internal error (userprintf.c): fp is 0.  Wrong qh_fprintf_rbox called.\n");
        qh_errexit_rbox(6231);
    }
    if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR)
      fprintf(fp, "QH%.4d ", msgcode);
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
} /* qh_fprintf_rbox */

