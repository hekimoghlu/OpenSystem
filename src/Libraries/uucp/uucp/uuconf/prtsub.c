/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
const char _uuconf_prtsub_rcsid[] = "$Id: prtsub.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

/* Clear the information in a port.  This can only clear the type
   independent information; the port type specific information is
   cleared when the type of the port is set.  */

void
_uuconf_uclear_port (qport)
     struct uuconf_port *qport;
{
  qport->uuconf_zname = NULL;
  qport->uuconf_ttype = UUCONF_PORTTYPE_UNKNOWN;
  qport->uuconf_zprotocols = NULL;
  qport->uuconf_qproto_params = NULL;

  /* Note that we do not set RELIABLE_SPECIFIED; this just sets
     defaults, so that ``seven-bit true'' does not imply ``reliable
     false''.  */
  qport->uuconf_ireliable = (UUCONF_RELIABLE_RELIABLE
			     | UUCONF_RELIABLE_EIGHT
			     | UUCONF_RELIABLE_FULLDUPLEX);

  qport->uuconf_zlockname = NULL;
  qport->uuconf_palloc = NULL;
}
