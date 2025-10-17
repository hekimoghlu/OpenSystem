/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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

#ifndef __KDC_LOCL_H__
#define __KDC_LOCL_H__

#include "headers.h"

typedef struct pk_client_params pk_client_params;
struct DigestREQ;
struct Kx509Request;
typedef struct kdc_request_desc *kdc_request_t;

#include <kdc-private.h>

#define FAST_EXPIRATION_TIME (3 * 60)

struct kdc_request_desc {
    krb5_context context;
    krb5_kdc_configuration *config;

    /* */

    krb5_data request;
    KDC_REQ req;
    METHOD_DATA *padata;

    /* out */

    METHOD_DATA outpadata;
    
    KDC_REP rep;
    EncTicketPart et;
    EncKDCRepPart ek;

    /* PA methods can affect both the reply key and the session key (pkinit) */
    krb5_enctype sessionetype;
    krb5_keyblock reply_key;
    krb5_keyblock session_key;

    const char *e_text;

    /* state */
    krb5_principal client_princ;
    char *client_name;
    hdb_entry_ex *client;
    HDB *clientdb;

    /* server used for krbtgt in TGS-REQ */
    krb5_principal server_princ;
    char *server_name;
    hdb_entry_ex *server;

    krb5_keyblock strengthen_key;

    /* only valid for tgs-req */
    krb5_enctype server_enctype;
    int rk_is_subkey;

    krb5_crypto armor_crypto;

    int use_fast_cookie;
    KDCFastState fast;
};


extern sig_atomic_t exit_flag;
extern size_t max_request_udp;
extern size_t max_request_tcp;
extern const char *request_log;
extern const char *port_str;
extern krb5_addresses explicit_addresses;

extern int enable_http;

#ifdef SUPPORT_DETACH

#define DETACH_IS_DEFAULT FALSE

extern int detach_from_console;
#endif

extern const struct units _kdc_digestunits[];

#define KDC_LOG_FILE		"kdc.log"
#ifdef __APPLE__
#define KDC_LOG_DIR		LOCALSTATEDIR "/log/krb5kdc"
#else
#define KDC_LOG_DIR		LOCALSTATEDIR "/log"
#endif

extern struct timeval _kdc_now;
#define kdc_time (_kdc_now.tv_sec)

extern char *runas_string;
extern char *chroot_string;
extern int listen_on_ipc;
extern int listen_on_network;

void
setup_listeners(krb5_context context, krb5_kdc_configuration *config, int, int);

krb5_kdc_configuration *
configure(krb5_context context, int argc, char **argv);

#ifdef __APPLE__
extern int sandbox_flag;
void bonjour_announce(heim_array_t (*get_realms)(void));
#endif

#ifdef HEIMDAL_PRINTF_ATTRIBUTE
#undef HEIMDAL_PRINTF_ATTRIBUTE
#define HEIMDAL_PRINTF_ATTRIBUTE(x)
#endif

#endif /* __KDC_LOCL_H__ */
