/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

#ifndef __KUSER_LOCL_H__
#define __KUSER_LOCL_H__

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
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

#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#include <roken.h>
#include <getarg.h>
#include <parse_time.h>
#include <err.h>
#include <krb5.h>

#if defined(HAVE_SYS_IOCTL_H) && SunOS != 40
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_IOCCOM_H
#include <sys/ioccom.h>
#endif
#ifndef NO_AFS
#include <kafs.h>
#endif
#include "crypto-headers.h" /* for UI_UTIL_read_pw_string */

#include <rtbl.h>

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif

#ifdef LIBINTL
#include <libintl.h>
#define N_(x,y) gettext(x)
#define NP_(x,y) (x)
#define getarg_i18n gettext
#else
#define N_(x,y) (x)
#define NP_(x,y) (x)
#define getarg_i18n NULL
#define bindtextdomain(package, localedir)
#define textdomain(package)
#endif

extern krb5_context kcc_context;

#endif /* __KUSER_LOCL_H__ */
