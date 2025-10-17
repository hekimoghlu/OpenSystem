/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
/* a *lot* of ugly global definitions that really should be removed...
 */

#include "telnetd.h"

RCSID("$Id$");

/*
 * Telnet server variable declarations
 */
char	options[256];
char	do_dont_resp[256];
char	will_wont_resp[256];
int	linemode;	/* linemode on/off */
int	flowmode;	/* current flow control state */
int	restartany;	/* restart output on any character state */
#ifdef DIAGNOSTICS
int	diagnostic;	/* telnet diagnostic capabilities */
#endif /* DIAGNOSTICS */
int	require_otp;

slcfun	slctab[NSLC + 1];	/* slc mapping table */

char	terminaltype[41];

/*
 * I/O data buffers, pointers, and counters.
 */
char	ptyobuf[BUFSIZ+NETSLOP], *pfrontp, *pbackp;

char	netibuf[BUFSIZ], *netip;

char	netobuf[BUFSIZ+NETSLOP], *nfrontp, *nbackp;
char	*neturg;		/* one past last bye of urgent data */

int	pcc, ncc;

int	ourpty, net;
int	SYNCHing;		/* we are in TELNET SYNCH mode */

/*
 * The following are some clocks used to decide how to interpret
 * the relationship between various variables.
 */

struct clocks_t clocks;


/* whether to log unauthenticated login attempts */
int log_unauth;

/* do not print warning if connection is not encrypted */
int no_warn;

/*
 * This function appends data to nfrontp and advances nfrontp.
 */

int
output_data (const char *format, ...)
{
  va_list args;
  int remaining, ret;

  va_start(args, format);
  remaining = BUFSIZ - (nfrontp - netobuf);
  ret = vsnprintf (nfrontp,
		   remaining,
		   format,
		   args);
  nfrontp += min(ret, remaining-1);
  va_end(args);
  return ret;
}
