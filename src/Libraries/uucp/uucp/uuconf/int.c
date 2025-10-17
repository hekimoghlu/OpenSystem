/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
const char _uuconf_int_rcsid[] = "$Id: int.c,v 1.9 2002/03/05 19:10:42 ian Rel $";
#endif

/* Parse a string into a variable.  This is called by uuconf_cmd_args,
   as well as other functions.  The parsing is done in a single place
   to make it easy to change.  This should return an error code,
   including both UUCONF_CMDTABRET_KEEP and UUCONF_CMDTABRET_EXIT if
   appropriate.  */

/*ARGSIGNORED*/
int
_uuconf_iint (qglobal, zval, p, fint)
     struct sglobal *qglobal ATTRIBUTE_UNUSED;
     const char *zval;
     pointer p;
     boolean fint;
{
  long i;
  char *zend;

  i = strtol ((char *) zval, &zend, 10);
  if (*zend != '\0')
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  if (fint)
    *(int *) p = (int) i;
  else
    *(long *) p = i;

  return UUCONF_CMDTABRET_CONTINUE;
}
