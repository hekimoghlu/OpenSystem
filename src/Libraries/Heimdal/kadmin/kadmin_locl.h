/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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

#ifndef __ADMIN_LOCL_H__
#define __ADMIN_LOCL_H__

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
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

#ifdef HAVE_UTIL_H
#include <util.h>
#endif
#ifdef HAVE_LIBUTIL_H
#include <libutil.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_SYS_UN_H
#include <sys/un.h>
#endif
#include <err.h>
#include <roken.h>
#include <krb5.h>
#include <krb5_locl.h>
#include <hdb.h>
#include <hdb_err.h>
#include <hex.h>
#include <kadm5/admin.h>
#include <kadm5/private.h>
#include <kadm5/kadm5_err.h>
#include <parse_time.h>
#include <getarg.h>

extern krb5_context context;
extern void * kadm_handle;

#undef ALLOC
#define ALLOC(X) ((X) = malloc(sizeof(*(X))))

/* util.c */

void attributes2str(krb5_flags, char *, size_t);
int  str2attributes(const char *, krb5_flags *);
int  parse_attributes (const char *, krb5_flags *, int *, int);
int  edit_attributes (const char *, krb5_flags *, int *, int);

int  parse_policy (const char *, char **, int *, int);
int  edit_policy (const char *, char **, int *, int);

void time_t2str(time_t, char *, size_t, int);
int  str2time_t (const char *, time_t *);
int  parse_timet (const char *, krb5_timestamp *, int *, int);
int  edit_timet (const char *, krb5_timestamp *, int *,
		 int);

void deltat2str(krb5_deltat, char *, size_t);
int  str2deltat(const char *, krb5_deltat *);
int  parse_deltat (const char *, krb5_deltat *, int *, int);
int  edit_deltat (const char *, krb5_deltat *, int *, int);

int edit_entry(kadm5_principal_ent_t, int *, kadm5_principal_ent_t, int);
void set_defaults(kadm5_principal_ent_t, int *, kadm5_principal_ent_t, int);
int set_entry(krb5_context, kadm5_principal_ent_t, int *,
	      const char *, const char *, const char *,
	      const char *, const char *, const char *);
int
foreach_principal(const char *, int (*)(krb5_principal, void*),
		  const char *, void *);

int parse_des_key (const char *, krb5_key_data *, const char **);

/* random_password.c */

void
random_password(char *, size_t);

/* kadm_conn.c */

extern sig_atomic_t term_flag, doing_useful_work;

void parse_ports(krb5_context, const char*);
void start_server(krb5_context, const char*);

/* server.c */

krb5_error_code
kadmind_loop (krb5_context, krb5_keytab, int);

/* rpc.c */

int
handle_mit(krb5_context, void *, size_t, int);


#endif /* __ADMIN_LOCL_H__ */
