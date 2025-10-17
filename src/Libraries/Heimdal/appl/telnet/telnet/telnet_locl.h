/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
/* $Id$ */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#include <errno.h>
#include <setjmp.h>
#ifdef HAVE_BSDSETJMP_H
#include <bsdsetjmp.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

/* termios.h *must* be included before curses.h, but not on Solaris 9,
   at least, where we end up with
   "/usr/include/term.h", line 1060: incomplete struct/union/enum termio: Ottyb
*/
#if defined HAVE_TERMIOS_H && !defined __sun
#include <termios.h>
#endif

#if defined(HAVE_CURSES_H)
#include <curses.h>
#ifdef HAVE_TERM_H
#include <term.h>
#endif
#elif defined(HAVE_TERMCAP_H)
#include <termcap.h>
#endif

#if defined(HAVE_SYS_TERMIO_H) && !defined(HAVE_TERMIOS_H)
#include <sys/termio.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif
#ifdef TIME_WITH_SYS_TIME
#include <sys/time.h>
#include <time.h>
#elif defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#else
#include <time.h>
#endif
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
/* not with SunOS 4 */
#if defined(HAVE_SYS_IOCTL_H) && SunOS != 40
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif /* HAVE_SYS_RESOURCE_H */
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif
#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif
#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#ifdef HAVE_NETINET6_IN6_H
#include <netinet6/in6.h>
#endif

#ifdef HAVE_NETINET_IN_SYSTM_H
#include <netinet/in_systm.h>
#endif
#ifdef HAVE_NETINET_IP_H
#include <netinet/ip.h>
#endif
#ifdef HAVE_ARPA_INET_H
#ifdef _AIX
struct sockaddr_dl; /* AIX fun */
struct ether_addr;
#endif
#include <arpa/inet.h>
#endif

#ifdef HAVE_ARPA_TELNET_H
#include <arpa/telnet.h>
#endif

#ifdef SOCKS
#include <socks.h>
#endif

#if	defined(AUTHENTICATION) || defined(ENCRYPTION)
#include <libtelnet/auth.h>
#include <libtelnet/encrypt.h>
#endif
#include <libtelnet/misc.h>
#include <libtelnet/misc-proto.h>

#define LINEMODE
#ifndef KLUDGELINEMODE
#define KLUDGELINEMODE
#endif

#include <err.h>
#include <roken.h>

#include "ring.h"
#include "externs.h"
#include "defines.h"
#include "types.h"

/* prototypes */

