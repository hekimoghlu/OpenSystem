/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
const char _uuconf_reliab_rcsid[] = "$Id: reliab.c,v 1.9 2002/03/05 19:10:42 ian Rel $";
#endif

/* Handle the "seven-bit" command for a port or a dialer.  The pvar
   argument points to an integer which should be set to hold
   reliability information.  */

/*ARGSUSED*/
int
_uuconf_iseven_bit (pglobal,argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int *pi = (int *) pvar;
  int fval;
  int iret;

  iret = _uuconf_iboolean (qglobal, argv[1], &fval);
  if ((iret &~ UUCONF_CMDTABRET_KEEP) != UUCONF_SUCCESS)
    return iret;

  *pi |= UUCONF_RELIABLE_SPECIFIED;
  if (fval)
    *pi &=~ UUCONF_RELIABLE_EIGHT;
  else
    *pi |= UUCONF_RELIABLE_EIGHT;

  return iret;
}

/* Handle the "reliable" command for a port or a dialer.  The pvar
   argument points to an integer which should be set to hold
   reliability information.  */

/*ARGSUSED*/
int
_uuconf_ireliable (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int *pi = (int *) pvar;
  int fval;
  int iret;

  iret = _uuconf_iboolean (qglobal, argv[1], &fval);
  if ((iret &~ UUCONF_CMDTABRET_KEEP) != UUCONF_SUCCESS)
    return iret;

  *pi |= UUCONF_RELIABLE_SPECIFIED;
  if (fval)
    *pi |= UUCONF_RELIABLE_RELIABLE;
  else
    *pi &=~ UUCONF_RELIABLE_RELIABLE;

  return iret;
}

/* Handle the "half-duplex" command for a port or a dialer.  The pvar
   argument points to an integer which should be set to hold
   reliability information.  */

/*ARGSUSED*/
int
_uuconf_ihalf_duplex (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc ATTRIBUTE_UNUSED;
     char **argv;
     pointer pvar;
     pointer pinfo ATTRIBUTE_UNUSED;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int *pi = (int *) pvar;
  int fval;
  int iret;

  iret = _uuconf_iboolean (qglobal, argv[1], &fval);
  if ((iret &~ UUCONF_CMDTABRET_KEEP) != UUCONF_SUCCESS)
    return iret;

  *pi |= UUCONF_RELIABLE_SPECIFIED;
  if (fval)
    *pi &=~ UUCONF_RELIABLE_FULLDUPLEX;
  else
    *pi |= UUCONF_RELIABLE_FULLDUPLEX;

  return iret;
}
