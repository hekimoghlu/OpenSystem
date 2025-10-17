/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
const char _uuconf_remunk_rcsid[] = "$Id: remunk.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

/* Get the name of the remote.unknown shell script.  */

/*ARGSUSED*/
int
uuconf_remote_unknown (pglobal, pzname)
     pointer pglobal ATTRIBUTE_UNUSED;
     char **pzname ATTRIBUTE_UNUSED;
{
#if ! HAVE_HDB_CONFIG
  return UUCONF_NOT_FOUND;
#else
#if HAVE_TAYLOR_CONFIG
  struct sglobal *qglobal = (struct sglobal *) pglobal;

  /* If ``unknown'' commands were used in the config file, then ignore
     any remote.unknown script.  */
  if (qglobal->qprocess->qunknown != NULL)
    return UUCONF_NOT_FOUND;
#endif /* HAVE_TAYLOR_CONFIG */

  return uuconf_hdb_remote_unknown (pglobal, pzname);
#endif /* HAVE_HDB_CONFIG */
}
