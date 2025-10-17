/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
const char _uuconf_val_rcsid[] = "$Id: val.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

/* Validate a login name for a system.  */

/*ARGSUSED*/
int
uuconf_validate (pglobal, qsys, zlogin)
     pointer pglobal;
     const struct uuconf_system *qsys;
     const char *zlogin;
{
#if HAVE_TAYLOR_CONFIG
  return uuconf_taylor_validate (pglobal, qsys, zlogin);
#else
  return UUCONF_SUCCESS;
#endif
}
