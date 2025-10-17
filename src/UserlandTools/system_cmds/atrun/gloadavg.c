/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#ifndef __APPLE__
#ifndef lint
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */
#endif //__APPLE__

#if !defined(__FreeBSD__) && !defined(__APPLE__)
#define _POSIX_SOURCE 1

/* System Headers */

#include <stdio.h>
#else
#include <stdlib.h>
#endif

/* Local headers */

#include "gloadavg.h"

/* Global functions */

void perr(const char *fmt, ...);

double
gloadavg(void)
/* return the current load average as a floating point number, or <0 for
 * error
 */
{
    double result;
#if !defined(__FreeBSD__) && !defined(__APPLE__)
    FILE *fp;
    
    if((fp=fopen(PROC_DIR "loadavg","r")) == NULL)
	result = -1.0;
    else
    {
	if(fscanf(fp,"%lf",&result) != 1)
	    result = -1.0;
	fclose(fp);
    }
#else
    if (getloadavg(&result, 1) != 1)
	    perr("error in getloadavg");
#endif
    return result;
}
