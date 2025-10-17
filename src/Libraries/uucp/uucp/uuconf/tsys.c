/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
const char _uuconf_tsys_rcsid[] = "$Id: tsys.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

/* Get system information from the Taylor UUCP configuration files.
   This is a wrapper for the internal function which makes sure that
   every field gets a default value.  */

int
uuconf_taylor_system_info (pglobal, zsystem, qsys)
     pointer pglobal;
     const char *zsystem;
     struct uuconf_system *qsys;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;

  iret = _uuconf_itaylor_system_internal (qglobal, zsystem, qsys);
  if (iret != UUCONF_SUCCESS)
    return iret;
  return _uuconf_isystem_basic_default (qglobal, qsys);
}
