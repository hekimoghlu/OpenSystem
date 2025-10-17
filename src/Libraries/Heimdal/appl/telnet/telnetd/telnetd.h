/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#include <config.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef TIME_WITH_SYS_TIME
#include <sys/time.h>
#include <time.h>
#elif defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#else
#include <time.h>
#endif

#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif /* HAVE_SYS_RESOURCE_H */

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

/* including both <sys/ioctl.h> and <termios.h> in SunOS 4 generates a
   lot of warnings */

#if defined(HAVE_SYS_IOCTL_H) && SunOS != 40
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
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

#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif

#include <signal.h>
#include <errno.h>
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#endif
#include <ctype.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <termios.h>

#ifdef HAVE_PTY_H
#include <pty.h>
#endif

#ifdef	STREAMSPTY
#ifdef HAVE_SAC_H
#include <sac.h>
#endif
#ifdef HAVE_SYS_STROPTS_H
#include <sys/stropts.h>
#endif

# include <stropts.h>

#ifdef  HAVE_SYS_UIO_H
#include <sys/uio.h>
#ifdef __hpux
#undef SE
#endif
#endif
#ifdef	HAVE_SYS_STREAM_H
#include <sys/stream.h>
#endif

#endif /* STREAMSPTY */

#undef NOERROR

#include "defs.h"

#ifndef _POSIX_VDISABLE
# ifdef VDISABLE
#  define _POSIX_VDISABLE VDISABLE
# else
#  define _POSIX_VDISABLE ((unsigned char)'\377')
# endif
#endif


#ifdef HAVE_SYS_PTY_H
#include <sys/pty.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif

#ifdef HAVE_SYS_PTYIO_H
#include <sys/ptyio.h>
#endif

#ifdef HAVE_SYS_UTSNAME_H
#include <sys/utsname.h>
#endif

#ifdef HAVE_PATHS_H
#include <paths.h>
#endif

#ifdef HAVE_ARPA_TELNET_H
#include <arpa/telnet.h>
#endif

#include "ext.h"

#ifdef SOCKS
#include <socks.h>
/* This doesn't belong here. */
struct tm *localtime(const time_t *);
struct hostent  *gethostbyname(const char *);
#endif

#ifdef AUTHENTICATION
#include <libtelnet/auth.h>
#include <libtelnet/misc.h>
#ifdef ENCRYPTION
#include <libtelnet/encrypt.h>
#endif
#endif

#ifdef HAVE_LIBUTIL_H
#include <libutil.h>
#endif

#include <roken.h>

/* Don't use the system login, use our version instead */

/* BINDIR should be defined somewhere else... */

#ifndef BINDIR
#define BINDIR "/usr/athena/bin"
#endif

#undef _PATH_LOGIN
#define _PATH_LOGIN	BINDIR "/login"

/* fallbacks */

#ifndef _PATH_DEV
#define _PATH_DEV "/dev/"
#endif

#ifndef _PATH_TTY
#define _PATH_TTY "/dev/tty"
#endif /* _PATH_TTY */

#ifdef	DIAGNOSTICS
#define	DIAG(a,b)	if (diagnostic & (a)) b
#else
#define	DIAG(a,b)
#endif

/* other external variables */
extern	char **environ;

/* prototypes */

/* appends data to nfrontp and advances */
int output_data (const char *format, ...)
#ifdef __GNUC__
__attribute__ ((format (printf, 1, 2)))
#endif
;

#ifdef ENCRYPTION
extern int require_encryption;
#endif
