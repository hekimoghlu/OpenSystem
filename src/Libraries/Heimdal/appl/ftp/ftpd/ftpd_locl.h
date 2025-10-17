/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#ifndef __ftpd_locl_h__
#define __ftpd_locl_h__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*
 * FTP server.
 */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#if defined(HAVE_SYS_IOCTL_H) && SunOS != 40
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_IOCCOM_H
#include <sys/ioccom.h>
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
#endif
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN_SYSTM_H
#include <netinet/in_systm.h>
#endif
#ifdef HAVE_NETINET_IP_H
#include <netinet/ip.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#include <arpa/ftp.h>
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_ARPA_TELNET_H
#include <arpa/telnet.h>
#endif

#include <ctype.h>
#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif
#include <errno.h>
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#include <glob.h>
#include <limits.h>
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#endif
#include <time.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_GRP_H
#include <grp.h>
#endif
#include <fnmatch.h>

#ifdef HAVE_BSD_BSD_H
#include <bsd/bsd.h>
#endif

#include <err.h>
#include "roken.h"

#include "pathnames.h"
#include "extern.h"
#include "common.h"

#include "security.h"

#ifdef KRB5
#include <krb5.h>
#endif /* KRB5 */

#if defined(KRB5)
#include <kafs.h>
#endif

#ifdef OTP
#include <otp.h>
#endif

#ifdef SOCKS
#include <socks.h>
extern int LIBPREFIX(fclose)      (FILE *);
#endif

/* SunOS doesn't have any declaration of fclose */

int fclose(FILE *stream);

int yyparse(void);

#ifndef LOG_FTP
#define LOG_FTP LOG_DAEMON
#endif

#endif /* __ftpd_locl_h__ */
