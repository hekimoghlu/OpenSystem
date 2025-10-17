/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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

#ifndef GSSAPI_KRB5_H_
#define GSSAPI_KRB5_H_

#include <gssapi.h>

GSSAPI_CPP_START

#if !defined(__GNUC__) && !defined(__attribute__)
#define __attribute__(x)
#endif

#ifndef GSSKRB5_FUNCTION_DEPRECATED
#define GSSKRB5_FUNCTION_DEPRECATED __attribute__((deprecated))
#endif


/*
 * This is for kerberos5 names.
 */

/* do not use this entry */
extern gss_OID_desc GSSAPI_LIB_VARIABLE __gss_krb5_nt_principal_oid_desc;
#define GSS_KRB5_NT_PRINCIPAL (&__gss_krb5_nt_principal_oid_desc)

extern gss_OID_desc GSSAPI_LIB_VARIABLE __gss_krb5_nt_principal_name_oid_desc;
#define GSS_KRB5_NT_PRINCIPAL_NAME (&__gss_krb5_nt_principal_name_oid_desc)

extern gss_OID_desc GSSAPI_LIB_VARIABLE __gss_krb5_nt_principal_name_referral_oid_desc;
#define GSS_KRB5_NT_PRINCIPAL_NAME_REFERRAL (&__gss_krb5_nt_principal_name_referral_oid_desc)

#define GSS_KRB5_NT_USER_NAME (&__gss_c_nt_user_name_oid_desc)
#define GSS_KRB5_NT_MACHINE_UID_NAME (&__gss_c_nt_machine_uid_name_oid_desc)
#define GSS_KRB5_NT_STRING_UID_NAME (&__gss_c_nt_string_uid_name_oid_desc)

/* for compatibility with MIT api */

#define gss_mech_krb5 GSS_KRB5_MECHANISM
#define gss_krb5_nt_general_name GSS_KRB5_NT_PRINCIPAL_NAME

/*
 * kerberos mechanism specific functions
 */

/*
 * Lucid - NFSv4 interface to GSS-API KRB5 to expose key material to
 * do GSS content token handling in-kernel.
 */

typedef struct gss_krb5_lucid_key {
	OM_uint32		type;
	OM_uint32		length;
	void * __nullable	data;
} gss_krb5_lucid_key_t;

typedef struct gss_krb5_rfc1964_keydata {
	OM_uint32		sign_alg;
	OM_uint32		seal_alg;
	gss_krb5_lucid_key_t	ctx_key;
} gss_krb5_rfc1964_keydata_t;

typedef struct gss_krb5_cfx_keydata {
	OM_uint32		have_acceptor_subkey;
	gss_krb5_lucid_key_t	ctx_key;
	gss_krb5_lucid_key_t	acceptor_subkey;
} gss_krb5_cfx_keydata_t;

typedef struct gss_krb5_lucid_context_v1 {
	OM_uint32	version;
	OM_uint32	initiate;
	OM_uint32	endtime;
	OM_uint64	send_seq;
	OM_uint64	recv_seq;
	OM_uint32	protocol;
	gss_krb5_rfc1964_keydata_t rfc1964_kd;
	gss_krb5_cfx_keydata_t	   cfx_kd;
} gss_krb5_lucid_context_v1_t;

typedef struct gss_krb5_lucid_context_version {
	OM_uint32	version;	/* Structure version number */
} gss_krb5_lucid_context_version_t;

GSSAPI_CPP_END

#endif /* GSSAPI_SPNEGO_H_ */
