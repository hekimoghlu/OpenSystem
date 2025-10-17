/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
 * $Id$
 */

#ifndef __HEADERS_H__
#define __HEADERS_H__

#include <config.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
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
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_UTIL_H
#include <util.h>
#endif
#ifdef HAVE_LIBUTIL_H
#include <libutil.h>
#endif
#include <err.h>
#include <roken.h>
#include <getarg.h>
#include <base64.h>
#include <parse_units.h>
#include <krb5.h>
#include <krb5_locl.h>
#ifdef DIGEST
#include <digest_asn1.h>
#endif
#ifdef KX509
#include <kx509_asn1.h>
#endif
#include <pku2u_asn1.h>
#include <hdb.h>
#include <hdb_err.h>
#include <der.h>

#ifndef NO_NTLM
#include <heimntlm.h>
#endif

#include <heim-ipc.h>
#include <kdc.h>
#include <windc_plugin.h>

#include <heimbase.h>

#undef ALLOC
#define ALLOC(X) ((X) = calloc(1, sizeof(*(X))))
#undef ALLOC_SEQ
#define ALLOC_SEQ(X, N) do { (X)->len = (N); \
(X)->val = calloc((X)->len, sizeof(*(X)->val)); } while(0)

#endif /* __HEADERS_H__ */
