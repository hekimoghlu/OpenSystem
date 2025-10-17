/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
/*
**
**  NAME:
**
**      LINUX/sysconf.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  This file contains all definitions specific to the Linux platform
**
**
*/

#ifndef _SYSCONF_H
#define _SYSCONF_H	1

/******************************************************************************/

#define DCETHREAD_USE_THROW
#define DCETHREAD_CHECKED
#include <dce/dcethread.h>
#include <dce/dce.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/file.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/time.h>

/*
*
* IMPORTANT ORDER DEPENDENCY:
*
*       The <cmaxxx.h> include files above do #ifdefs on
*       _POSIX_REENTRANT_FUNCTIONS.  Under OSF/1, this
*       is set in <unistd.h>.  The problem is that for
*       the OSF/1 DCE reference port, we need to build
*       with vanilla CMA threads and no help from the
*       kernel.
*
*       Therefore, the include of <unistd.h> must come
*       after the <cmaxxx.h> files above.
*
*       You have been warned.
*
*/

#include <unistd.h>

#include <assert.h>
#include <fcntl.h>
#include <string.h>

/*#define NO_VOID_STAR	1*/	/* void * is supported in GCC -ansi mode*/

#define USE_PROTOTYPES  1

#define STDARG_PRINTF   1

#define NO_VARARGS_PRINTF 1

#define BSD_4_4_SOCKET 1

/* if SOCKADDR_LEN isn't defined MSG_MAXIOVLEN will not be defined in
 * <sys/socket.h>
 */

#ifndef MSG_MAXIOVLEN
#define MSG_MAXIOVLEN	UIO_MAXIOV
#endif /* MSG_MAXIOVLEN */

/**************************************************************************/

/*
 * This definition means the ioctl() call to get the interface
 * addresses in ipnaf_bsd.c in enumerate_interfaces() will not be
 * called. This is because Ultrix returns the same internet address
 * for all interfaces.
 */

#define NO_SIOCGIFADDR 1

/*
 * Define protocol sequences that are always available on
 * OSF/1 platforms
 */

#ifndef PROT_NCACN
#define PROT_NCACN	1
#endif

#ifndef PROT_NCADG
#define PROT_NCADG	1
#endif

#ifndef NAF_IP
#define NAF_IP	1
#endif

#define RPC_DEFAULT_NLSPATH "/usr/lib/nls/msg/en_US.ISO8859-1/%s.cat"

/* From /usr/include/linux/un.h UNIX_PATH_MAX */
#define RPC_C_PATH_NP_MAX	108

/****************************************************************************/

/* define some macros to support atfork handler */

#define ATFORK_SUPPORTED

#define ATFORK(handler) rpc__atfork(handler)

extern void rpc__cma_atfork (void *);

/****************************************************************************/

#endif /* _SYSCONF_H */
