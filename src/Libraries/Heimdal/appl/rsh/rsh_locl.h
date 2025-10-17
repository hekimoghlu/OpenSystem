/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#include <assert.h>
#include <stdarg.h>
#include <ctype.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
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

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#ifdef HAVE_SHADOW_H
#include <shadow.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#include <errno.h>

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#endif
#ifdef HAVE_PATHS_H
#include <paths.h>
#endif
#include <err.h>
#include <roken.h>
#include <getarg.h>
#ifdef KRB5
#include <krb5.h>
/* XXX */
struct hx509_certs_data;
struct krb5_pk_identity;
struct krb5_pk_cert;
struct ContentInfo;
struct AlgorithmIdentifier;
struct _krb5_krb_auth_data;
struct krb5_dh_moduli;
struct _krb5_key_data;
struct _krb5_encryption_type;
struct _krb5_srv_query_ctx;
struct _krb5_key_type;
#include "crypto-headers.h"
#include <krb5-private.h> /* for _krb5_{get,put}_int */
#endif
#if defined(KRB5)
#include <kafs.h>
#endif

#ifndef _PATH_BSHELL
#define _PATH_BSHELL	"/bin/sh"
#endif

#ifndef _PATH_DEFPATH
#define _PATH_DEFPATH	"/usr/bin:/bin"
#endif

#include "loginpaths.h"

/*
 *
 */

enum auth_method { AUTH_KRB5, AUTH_BROKEN };

extern enum auth_method auth_method;
extern int do_encrypt;
#ifdef KRB5
extern krb5_context context;
extern krb5_keyblock *keyblock;
extern krb5_crypto crypto;
extern int key_usage;
extern void *ivec_in[2];
extern void *ivec_out[2];
void init_ivecs(int, int);
#endif

#define KCMD_OLD_VERSION "KCMDV0.1"
#define KCMD_NEW_VERSION "KCMDV0.2"

#define USERNAME_SZ 16
#ifndef ARG_MAX
#define ARG_MAX 8192
#endif

#define RSH_BUFSIZ (5 * 1024) /* MIT kcmd can't handle larger buffers */
#define RSHD_BUFSIZ (16 * 1024) /* Old maxize for Heimdal 0.4 rsh */

#define PATH_RSH BINDIR "/rsh"

#if defined(KRB5)
ssize_t do_read (int, void*, size_t, void*);
ssize_t do_write (int, void*, size_t, void*);
#else
#define do_write(F, B, L, I) write((F), (B), (L))
#define do_read(F, B, L, I) read((F), (B), (L))
#endif
