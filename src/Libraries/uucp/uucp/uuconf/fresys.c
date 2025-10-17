/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
const char _uuconf_fresys_rcsid[] = "$Id: fresys.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

/* Free the memory allocated for a system.  */

#undef uuconf_system_free

/*ARGSUSED*/
int
uuconf_system_free (pglobal, qsys)
     pointer pglobal ATTRIBUTE_UNUSED;
     struct uuconf_system *qsys;
{
  uuconf_free_block (qsys->uuconf_palloc);
  return UUCONF_SUCCESS;
}
