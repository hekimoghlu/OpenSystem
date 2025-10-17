/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

#include <ctype.h>

#include "uudefs.h"

/* See whether a file is a spool file.  Spool file names are specially
   crafted to hand around to other UUCP packages.  They always begin
   with 'C', 'D' or 'X', and the second character is always a period.
   The remaining characters may be any printable characters, since
   they may include a grade set by another system.  */

boolean
fspool_file (zfile)
     const char *zfile;
{
  const char *z;

  if (*zfile != 'C' && *zfile != 'D' && *zfile != 'X')
    return FALSE;
  if (zfile[1] != '.')
    return FALSE;
  for (z = zfile + 2; *z != '\0'; z++)
    if (*z == '/' || ! isprint (BUCHAR (*z)) || isspace (BUCHAR (*z)))
      return FALSE;
  return TRUE;
}
