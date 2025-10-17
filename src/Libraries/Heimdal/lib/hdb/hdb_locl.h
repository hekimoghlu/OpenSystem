/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#ifndef __HDB_LOCL_H__
#define __HDB_LOCL_H__

#include <assert.h>
#include <heimbase.h>

#include <config.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#include <roken.h>

struct krb5_dh_moduli;
struct AlgorithmIdentifier;
struct _krb5_krb_auth_data;
struct _krb5_key_data;
struct _krb5_key_type;
struct _krb5_checksum_type;
struct _krb5_encryption_type;
struct _krb5_srv_query_ctx;
struct krb5_fast_state;
struct _krb5_srp_group;
struct _krb5_srp;

#include "crypto-headers.h"
#include <heimbase.h>
#include <hx509.h>
#include <krb5.h>
#include <krb5-private.h>
#include <hdb.h>
#include <hdb-private.h>

krb5_error_code hdb_od_create(krb5_context, HDB **, const char *);
krb5_error_code hdb_keytab_create(krb5_context, HDB **, const char *);
krb5_error_code _hdb_keytab2hdb_entry(krb5_context, const krb5_keytab_entry *, hdb_entry_ex *);

#ifdef __APPLE__
#define HDB_DEFAULT_DB "od:/Local/Default"
#else
#define HDB_DEFAULT_DB HDB_DB_DIR "/heimdal"
#endif

#define HDB_DB_FORMAT_ENTRY "hdb/db-format"

#define KRB5_KDB_DISALLOW_POSTDATED	0x00000001
#define KRB5_KDB_DISALLOW_FORWARDABLE	0x00000002
#define KRB5_KDB_DISALLOW_TGT_BASED	0x00000004
#define KRB5_KDB_DISALLOW_RENEWABLE	0x00000008
#define KRB5_KDB_DISALLOW_PROXIABLE	0x00000010
#define KRB5_KDB_DISALLOW_DUP_SKEY	0x00000020
#define KRB5_KDB_DISALLOW_ALL_TIX	0x00000040
#define KRB5_KDB_REQUIRES_PRE_AUTH	0x00000080
#define KRB5_KDB_REQUIRES_HW_AUTH	0x00000100
#define KRB5_KDB_REQUIRES_PWCHANGE	0x00000200
#define KRB5_KDB_DISALLOW_SVR		0x00001000
#define KRB5_KDB_PWCHANGE_SERVICE	0x00002000
#define KRB5_KDB_SUPPORT_DESMD5		0x00004000
#define KRB5_KDB_NEW_PRINC		0x00008000

#undef ALLOC
#define ALLOC(X) ((X) = calloc(1, sizeof(*(X))))
#undef ALLOC_SEQ
#define ALLOC_SEQ(X, N) do { (X)->len = (N); \
(X)->val = calloc((X)->len, sizeof(*(X)->val)); } while(0)

#endif /* __HDB_LOCL_H__ */
