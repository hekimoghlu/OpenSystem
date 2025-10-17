/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#include "sysdep.h"
#include "system.h"

#if HAVE_TCP
#if HAVE_SYS_TYPES_TCP_H
#include <sys/types.tcp.h>
#endif
#include <sys/socket.h>
#endif

#ifndef ttyname
extern char *ttyname ();
#endif

/* Get the port name of standard input.  I assume that Unix systems
   generally support ttyname.  If they don't, this function can just
   return NULL.  It uses getsockname to see whether standard input is
   a TCP connection.  */

const char *
zsysdep_port_name (ftcp_port)
     boolean *ftcp_port;
{
  const char *z;

  *ftcp_port = FALSE;

#if HAVE_TCP
  {
    size_t clen;
    struct sockaddr s;

    clen = sizeof (struct sockaddr);
    if (getsockname (0, &s, &clen) == 0)
      *ftcp_port = TRUE;
  }
#endif /* HAVE_TCP */

  z = ttyname (0);
  if (z == NULL)
    return NULL;
  if (strncmp (z, "/dev/", sizeof "/dev/" - 1) == 0)
    return z + sizeof "/dev/" - 1;
  else
    return z;
}
