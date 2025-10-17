/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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

#ifndef __kadm5_privatex_h__
#define __kadm5_privatex_h__

#include <gssapi.h>

struct kadm_func {
    kadm5_ret_t (*chpass_principal) (void *, krb5_principal, int, const char*, int, krb5_key_salt_tuple *);
    kadm5_ret_t (*create_principal) (void*, kadm5_principal_ent_t,
				     uint32_t, const char*, int, krb5_key_salt_tuple *);
    kadm5_ret_t (*delete_principal) (void*, krb5_principal);
    kadm5_ret_t (*destroy) (void*);
    kadm5_ret_t (*flush) (void*);
    kadm5_ret_t (*get_principal) (void*, krb5_principal,
				  kadm5_principal_ent_t, uint32_t);
    kadm5_ret_t (*get_principals) (void*, const char*, char***, int*);
    kadm5_ret_t (*get_privs) (void*, uint32_t*);
    kadm5_ret_t (*modify_principal) (void*, kadm5_principal_ent_t, uint32_t);
    kadm5_ret_t (*randkey_principal) (void*, krb5_principal, krb5_boolean, int,
				      krb5_key_salt_tuple*, krb5_keyblock**,
				      int*);
    kadm5_ret_t (*rename_principal) (void*, krb5_principal, krb5_principal);
    kadm5_ret_t (*chpass_principal_with_key) (void *, krb5_principal, int,
					      int, krb5_key_data *);
    kadm5_ret_t (*lock) (void *);
    kadm5_ret_t (*unlock) (void *);
};

/* XXX should be integrated */
typedef struct kadm5_common_context {
    krb5_context context;
    krb5_boolean my_context;
    struct kadm_func funcs;
    void *data;
}kadm5_common_context;

typedef struct kadm5_log_peer {
    int fd;
    char *name;
    krb5_auth_context ac;
    struct kadm5_log_peer *next;
} kadm5_log_peer;

typedef struct kadm5_log_context {
    char *log_file;
    int log_fd;
    uint32_t version;
#ifndef NO_UNIX_SOCKETS
    struct sockaddr_un socket_name;
#else
    struct addrinfo *socket_info;
#endif
    krb5_socket_t socket_fd;
} kadm5_log_context;

typedef struct kadm5_server_context {
    krb5_context context;
    krb5_boolean my_context;
    struct kadm_func funcs;
    /* */
    kadm5_config_params config;
    HDB *db;
    int keep_open;
    krb5_principal caller;
    unsigned acl_flags;
    kadm5_log_context log_context;
} kadm5_server_context;

typedef struct kadm5_client_context {
    krb5_context context;
    krb5_boolean my_context;
    struct kadm_func funcs;
    /* */
    krb5_auth_context ac;
    char *realm;
    char *admin_server;
    int kadmind_port;
    int sock;
    char *client_name;
    char *service_name;
    krb5_prompter_fct prompter;
    const char *keytab;
    krb5_ccache ccache;
    kadm5_config_params *realm_params;
}kadm5_client_context;

typedef struct kadm5_ad_context {
    krb5_context context;
    krb5_boolean my_context;
    struct kadm_func funcs;
    /* */
    kadm5_config_params config;
    krb5_principal caller;
    krb5_ccache ccache;
    char *client_name;
    char *realm;
    void *ldap_conn;
    char *base_dn;
} kadm5_ad_context;

typedef struct kadm5_mit_context {
    krb5_context context;
    krb5_boolean my_context;
    struct kadm_func funcs;
    /* */
    char *admin_server;
    char *realm;
    int kadmind_port;
    kadm5_config_params config;
    krb5_principal caller;
    void *gsscontext;
} kadm5_mit_context;

enum kadm_ops {
    kadm_get,
    kadm_delete,
    kadm_create,
    kadm_rename,
    kadm_chpass,
    kadm_modify,
    kadm_randkey,
    kadm_get_privs,
    kadm_get_princs,
    kadm_chpass_with_key,
    kadm_nop
};

#define KADMIN_APPL_VERSION "KADM0.1"
#define KADMIN_OLD_APPL_VERSION "KADM0.0"

struct _kadm5_xdr_opaque_auth {
    uint32_t flavor;
    krb5_data data;
};

struct _kadm5_xdr_call_header {
    uint32_t xid;
    uint32_t rpcvers;
    uint32_t prog;
    uint32_t vers;
    uint32_t proc;
    struct _kadm5_xdr_opaque_auth cred;
    krb5_data headercopy;
    struct _kadm5_xdr_opaque_auth verf;
};

struct _kadm5_xdr_gcred {
    uint32_t version;
    uint32_t proc;
    uint32_t seq_num;
    uint32_t service;
    krb5_data handle;
};

struct _kadm5_xdr_gacred {
    uint32_t version;
    uint32_t auth_msg;
    krb5_data handle;
};

#include "kadm5-private.h"

#endif /* __kadm5_privatex_h__ */
