/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include <string.h>

#define ARRAYSIZE 128

main (argc, argv)
int argc;
char *argv[];
{
  char line[ARRAYSIZE];

  do {
    memset (line, 0, ARRAYSIZE);
    fgets (line, ARRAYSIZE, stdin);
    *(line + strlen(line)-1) = '\0'; /* get rid of the newline */

    /* look for a few simple commands */
    if (strncmp (line,"prompt ", 6) == 0) {
      printf ("%s (y or n) ?", line + 6);
      if (getchar() == 'y')
	puts ("YES");
      else
	puts ("NO");
    }
    if (strncmp (line, "print ", 6) == 0) {
      puts (line + 6);
    }
  } while (strncmp (line, "quit", 4));
}
