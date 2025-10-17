/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "uucnfi.h"

#if USE_RCS_ID
const char _uuconf_strip_rcsid[] = "$Id: strip.c,v 1.3 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get information about what types of global information should be
   stripped.  There are currently only two, which we return as a
   couple of bits.  We store them as two separate variables, so we
   don't need to have a special function to set the values correctly.  */

int
uuconf_strip (pglobal, pistrip)
     pointer pglobal;
     int *pistrip;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;

  *pistrip = 0;
  if (qglobal->qprocess->fstrip_login)
    *pistrip |= UUCONF_STRIP_LOGIN;
  if (qglobal->qprocess->fstrip_proto)
    *pistrip |= UUCONF_STRIP_PROTO;
  return UUCONF_SUCCESS;
}
