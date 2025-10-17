/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
/*
 * Include necessary headers...
 */

#include <stdio.h>
#include "cups.h"
#include <sys/time.h>


/*
 * 'main()' - Loop calling cupsGetDests.
 */

int                                     /* O - Exit status */
main(void)
{
  int           num_dests;              /* Number of destinations */
  cups_dest_t   *dests;                 /* Destinations */
  struct timeval start, end;            /* Start and stop time */
  double        secs;                   /* Total seconds to run cupsGetDests */


  for (;;)
  {
    gettimeofday(&start, NULL);
    num_dests = cupsGetDests(&dests);
    gettimeofday(&end, NULL);
    secs = end.tv_sec - start.tv_sec + 0.000001 * (end.tv_usec - start.tv_usec);

    printf("Found %d printers in %.3f seconds...\n", num_dests, secs);

    cupsFreeDests(num_dests, dests);
    sleep(1);
  }

  return (0);
}
