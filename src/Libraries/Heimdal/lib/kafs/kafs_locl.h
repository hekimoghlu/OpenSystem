/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

#ifndef __KAFS_LOCL_H__
#define __KAFS_LOCL_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <errno.h>

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#if defined(HAVE_SYS_IOCTL_H) && SunOS != 40
#include <sys/ioctl.h>
#endif
#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif
#ifdef HAVE_SYS_SYSCTL_H
#include <sys/sysctl.h>
#endif

#ifdef HAVE_SYS_SYSCALL_H
#include <sys/syscall.h>
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

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#ifdef HAVE_ARPA_NAMESER_H
#include <arpa/nameser.h>
#endif
#ifdef HAVE_RESOLV_H
#include <resolv.h>
#endif
#include <roken.h>

#ifdef KRB5
#include <krb5.h>
#endif
#ifdef KRB5
#include "crypto-headers.h"
#include <krb5-v4compat.h>
typedef struct credentials CREDENTIALS;
#endif /* KRB5 */
#include <kafs.h>

#include <resolve.h>

#include "afssysdefs.h"

struct kafs_data;
struct kafs_token;
typedef int (*afslog_uid_func_t)(struct kafs_data *,
				 const char *,
				 const char *,
				 uid_t,
				 const char *);

typedef int (*get_cred_func_t)(struct kafs_data*, const char*, const char*,
			       const char*, uid_t, struct kafs_token *);

typedef char* (*get_realm_func_t)(struct kafs_data*, const char*);

struct kafs_data {
    const char *name;
    afslog_uid_func_t afslog_uid;
    get_cred_func_t get_cred;
    get_realm_func_t get_realm;
    const char *(*get_error)(struct kafs_data *, int);
    void (*free_error)(struct kafs_data *, const char *);
    void *data;
};

struct kafs_token {
    struct ClearToken ct;
    void *ticket;
    size_t ticket_len;
};

void _kafs_foldup(char *, const char *);

int _kafs_afslog_all_local_cells(struct kafs_data*, uid_t, const char*);

int _kafs_get_cred(struct kafs_data*, const char*, const char*, const char *,
		   uid_t, struct kafs_token *);

int
_kafs_realm_of_cell(struct kafs_data *, const char *, char **);

int
_kafs_v4_to_kt(CREDENTIALS *, uid_t, struct kafs_token *);

void
_kafs_fixup_viceid(struct ClearToken *, uid_t);

#ifdef _AIX
int aix_pioctl(char*, int, struct ViceIoctl*, int);
int aix_setpag(void);
#endif

#endif /* __KAFS_LOCL_H__ */
