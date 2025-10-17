/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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

#ifndef NTLM_NTLM_H
#define NTLM_NTLM_H

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include <roken.h>

#include <gssapi.h>
#include <gssapi_ntlm.h>
#include <gssapi_spi.h>
#include <gssapi_mech.h>
#include <gssapi_oid.h>

#include <krb5.h>
#include <heim_threads.h>

#include <kcm.h>
#include <hex.h>

#include <heimntlm.h>

#define HC_DEPRECATED_CRYPTO
#include "crypto-headers.h"

typedef struct {
    char *user;
    char *domain;
    int flags;
#define NTLM_UUID	1
#define NTLM_ANON_NAME	2
#define NTLM_DS_UUID	4
    unsigned char ds_uuid[16];
    unsigned char uuid[16];
} ntlm_name_desc, *ntlm_name;

struct ntlm_ctx;

typedef ntlm_name ntlm_cred;

typedef OM_uint32
(*ntlm_interface_init)(OM_uint32 *, void **);

typedef OM_uint32
(*ntlm_interface_destroy)(OM_uint32 *, void *);

typedef int
(*ntlm_interface_probe)(OM_uint32 *, void *, const char *, unsigned int *flags);

typedef OM_uint32
(*ntlm_interface_type3)(OM_uint32 *, struct ntlm_ctx *, void *, const struct ntlm_type3 *,
			ntlm_cred, uint32_t *, uint32_t *, struct ntlm_buf *,
			ntlm_name *, struct ntlm_buf *, struct ntlm_buf *);

typedef OM_uint32
(*ntlm_interface_targetinfo)(OM_uint32 *,
			     struct ntlm_ctx *,
			     void *,
			     const char *,
			     const char *,
			     uint32_t *);


typedef void
(*ntlm_interface_free_buffer)(struct ntlm_buf *);

struct ntlm_server_interface {
    const char *nsi_name;
    ntlm_interface_init nsi_init;
    ntlm_interface_destroy nsi_destroy;
    ntlm_interface_probe nsi_probe;
    ntlm_interface_type3 nsi_type3;
    ntlm_interface_free_buffer nsi_free_buffer;
    ntlm_interface_targetinfo nsi_ti;
};


struct ntlmv2_key {
    uint32_t seq;
    EVP_CIPHER_CTX sealkey;
    EVP_CIPHER_CTX *signsealkey;
    unsigned char signkey[16];
};

extern struct ntlm_server_interface ntlmsspi_kdc_digest;
extern struct ntlm_server_interface ntlmsspi_dstg_digest;
extern struct ntlm_server_interface ntlmsspi_netr_digest;
extern struct ntlm_server_interface ntlmsspi_od_digest;


struct ntlm_backend {
    struct ntlm_server_interface *interface;
    void *ctx;
};


typedef struct ntlm_ctx {
    struct ntlm_backend *backends;
    size_t num_backends;
    ntlm_cred client;

    unsigned int probe_flags;
#define NSI_NO_SIGNING 1

    OM_uint32 gssflags;
    uint32_t kcmflags;
    uint32_t flags;
    uint32_t status;
#define STATUS_OPEN 1
#define STATUS_CLIENT 2
#define STATUS_SESSIONKEY 4
    krb5_data sessionkey;
    krb5_data type1;
    krb5_data type2;
    krb5_data type3;

    uint8_t challenge[8];

    struct ntlm_targetinfo ti;
    struct ntlm_buf targetinfo;

    gss_name_t srcname;
    gss_name_t targetname;
    char *clientsuppliedtargetname;

    char uuid[16];
    gss_buffer_desc pac;

    union {
	struct {
	    struct {
		uint32_t seq;
		EVP_CIPHER_CTX key;
	    } crypto_send, crypto_recv;
	} v1;
	struct {
	    struct ntlmv2_key send, recv;
	} v2;
    } u;
} *ntlm_ctx;

#include <ntlm-private.h>


#endif /* NTLM_NTLM_H */
