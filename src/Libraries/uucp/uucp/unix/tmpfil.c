/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

#include "uudefs.h"
#include "uuconf.h"
#include "system.h"
#include "sysdep.h"

#define ZDIGS \
  "0123456789abcdefghijklmnopqrtsuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
#define CDIGS (sizeof ZDIGS - 1)

/*ARGSUSED*/
char *
zstemp_file (qsys)
     const struct uuconf_system *qsys ATTRIBUTE_UNUSED;
{
  static unsigned int icount;
  const char *const zdigs = ZDIGS;
  char ab[14];
  pid_t ime;
  int iset;

  ab[0] = 'T';
  ab[1] = 'M';
  ab[2] = '.';

  ime = getpid ();
  iset = 3;
  while (ime > 0 && iset < 10)
    {
      ab[iset] = zdigs[ime % CDIGS];
      ime /= CDIGS;
      ++iset;
    }

  ab[iset] = '.';
  ++iset;

  ab[iset] = zdigs[icount / CDIGS];
  ++iset;
  ab[iset] = zdigs[icount % CDIGS];
  ++iset;

  ab[iset] = '\0';

  ++icount;
  if (icount >= CDIGS * CDIGS)
    icount = 0;

#if SPOOLDIR_V2 || SPOOLDIR_BSD42
  return zbufcpy (ab);
#endif
#if SPOOLDIR_BSD43 || SPOOLDIR_ULTRIX || SPOOLDIR_TAYLOR
  return zsysdep_in_dir (".Temp", ab);
#endif
#if SPOOLDIR_HDB || SPOOLDIR_SVR4
  return zsysdep_in_dir (qsys->uuconf_zname, ab);
#endif
}
