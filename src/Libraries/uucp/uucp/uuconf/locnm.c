/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
const char _uuconf_locnm_rcsid[] = "$Id: locnm.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get the local node name.  */

int
uuconf_localname (pglobal, pzname)
     pointer pglobal;
     const char **pzname;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;

  *pzname = qglobal->qprocess->zlocalname;
  if (*pzname != NULL)
    return UUCONF_SUCCESS;
  else
    return UUCONF_NOT_FOUND;
}
