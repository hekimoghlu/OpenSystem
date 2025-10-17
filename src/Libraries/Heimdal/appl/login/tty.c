/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include "login_locl.h"

RCSID("$Id$");

/*
 * Clean the tty name.  Return a pointer to the cleaned version.
 */

char *
clean_ttyname (char *tty)
{
  char *res = tty;

  if (strncmp (res, _PATH_DEV, strlen(_PATH_DEV)) == 0)
    res += strlen(_PATH_DEV);
  if (strncmp (res, "pty/", 4) == 0)
    res += 4;
  if (strncmp (res, "ptym/", 5) == 0)
    res += 5;
  return res;
}

/*
 * Generate a name usable as an `ut_id', typically without `tty'.
 */

char *
make_id (char *tty)
{
  char *res = tty;

  if (strncmp (res, "pts/", 4) == 0)
    res += 4;
  if (strncmp (res, "tty", 3) == 0)
    res += 3;
  return res;
}
