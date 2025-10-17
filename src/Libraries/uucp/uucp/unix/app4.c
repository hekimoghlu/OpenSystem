/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#include "uucp.h"

#include "uudefs.h"
#include "sysdep.h"

char *
zsappend4 (zdir1, zdir2, zdir3, zfile)
     const char *zdir1;
     const char *zdir2;
     const char *zdir3;
     const char *zfile;
{
  size_t cdir1, cdir2, cdir3, cfile;
  char *zret;

  cdir1 = strlen (zdir1);
  cdir2 = strlen (zdir2);
  cdir3 = strlen (zdir3);
  cfile = strlen (zfile);
  zret = zbufalc (cdir1 + cdir2 + cdir3 + cfile + 4);
  if (cdir1 == 1 && *zdir1 == '/')
    cdir1 = 0;
  else
    memcpy (zret, zdir1, cdir1);
  memcpy (zret + cdir1 + 1, zdir2, cdir2);
  memcpy (zret + cdir1 + cdir2 + 2, zdir3, cdir3);
  memcpy (zret + cdir1 + cdir2 + cdir3 + 3, zfile, cfile);
  zret[cdir1] = '/';
  zret[cdir1 + cdir2 + 1] = '/';
  zret[cdir1 + cdir2 + cdir3 + 2] = '/';
  zret[cdir1 + cdir2 + cdir3 + cfile + 3] = '\0';
  return zret;
}
