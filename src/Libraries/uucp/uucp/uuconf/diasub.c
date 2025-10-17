/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
const char _uuconf_diasub_rcsid[] = "$Id: diasub.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

/* Clear the information in a dialer.  */

#define INIT_CHAT(q) \
  ((q)->uuconf_pzchat = NULL, \
   (q)->uuconf_pzprogram = NULL, \
   (q)->uuconf_ctimeout = 60, \
   (q)->uuconf_pzfail = NULL, \
   (q)->uuconf_fstrip = TRUE)

void
_uuconf_uclear_dialer (qdialer)
     struct uuconf_dialer *qdialer;
{
  qdialer->uuconf_zname = NULL;
  INIT_CHAT (&qdialer->uuconf_schat);
  qdialer->uuconf_zdialtone = (char *) ",";
  qdialer->uuconf_zpause = (char *) ",";
  qdialer->uuconf_fcarrier = TRUE;
  qdialer->uuconf_ccarrier_wait = 60;
  qdialer->uuconf_fdtr_toggle = FALSE;
  qdialer->uuconf_fdtr_toggle_wait = FALSE;
  INIT_CHAT (&qdialer->uuconf_scomplete);
  INIT_CHAT (&qdialer->uuconf_sabort);
  qdialer->uuconf_qproto_params = NULL;
  /* Note that we do not set RELIABLE_SPECIFIED; this just sets
     defaults, so that ``seven-bit true'' does not imply ``reliable
     false''.  */
  qdialer->uuconf_ireliable = (UUCONF_RELIABLE_RELIABLE
			       | UUCONF_RELIABLE_EIGHT
			       | UUCONF_RELIABLE_FULLDUPLEX);
  qdialer->uuconf_palloc = NULL;
}
