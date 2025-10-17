/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
const char _uuconf_base_rcsid[] = "$Id: base.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

/* This turns a cmdtab_offset table into a uuconf_cmdtab table.  Each
   offset is adjusted by a base value.  */

void
_uuconf_ucmdtab_base (qoff, celes, pbase, qset)
     register const struct cmdtab_offset *qoff;
     size_t celes;
     char *pbase;
     register struct uuconf_cmdtab *qset;
{
  register size_t i;

  for (i = 0; i < celes; i++, qoff++, qset++)
    {
      qset->uuconf_zcmd = qoff->zcmd;
      qset->uuconf_itype = qoff->itype;
      if (qoff->ioff == (size_t) -1)
	qset->uuconf_pvar = NULL;
      else
	qset->uuconf_pvar = pbase + qoff->ioff;
      qset->uuconf_pifn = qoff->pifn;
    }
}
