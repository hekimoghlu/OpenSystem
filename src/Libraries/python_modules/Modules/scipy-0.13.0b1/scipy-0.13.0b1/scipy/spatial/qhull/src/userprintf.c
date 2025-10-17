/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
   >-------------------------------</a><a name="qh_fprintf">-</a>

   qh_fprintf(fp, msgcode, format, list of args )
     print arguments to *fp according to format
     Use qh_fprintf_rbox() for rboxlib.c

   notes:
     same as fprintf()
     fgets() is not trapped like fprintf()
     exit qh_fprintf via qh_errexit()
*/

void qh_fprintf(FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;

    if (!fp) {
        fprintf(stderr, "QH6232 Qhull internal error (userprintf.c): fp is 0.  Wrong qh_fprintf called.\n");
        qh_errexit(6232, NULL, NULL);
    }
    va_start(args, fmt);
#if qh_QHpointer
    if (qh_qh && qh ANNOTATEoutput) {
#else
    if (qh ANNOTATEoutput) {
#endif
      fprintf(fp, "[QH%.4d]", msgcode);
    }else if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR ) {
      fprintf(fp, "QH%.4d ", msgcode);
    }
    vfprintf(fp, fmt, args);
    va_end(args);

    /* Place debugging traps here. Use with option 'Tn' */

} /* qh_fprintf */

