/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
const char _uuconf_dial_rcsid[] = "$Id: dial.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

/* Find a dialer by name.  */

int
uuconf_dialer_info (pglobal, zdialer, qdialer)
     pointer pglobal;
     const char *zdialer;
     struct uuconf_dialer *qdialer;
{
#if HAVE_HDB_CONFIG
  struct sglobal *qglobal = (struct sglobal *) pglobal;
#endif
  int iret;

#if HAVE_TAYLOR_CONFIG
  iret = uuconf_taylor_dialer_info (pglobal, zdialer, qdialer);
  if (iret != UUCONF_NOT_FOUND)
    return iret;
#endif

#if HAVE_HDB_CONFIG
  if (qglobal->qprocess->fhdb)
    {
      iret = uuconf_hdb_dialer_info (pglobal, zdialer, qdialer);
      if (iret != UUCONF_NOT_FOUND)
	return iret;
    }
#endif

  return UUCONF_NOT_FOUND;
}
