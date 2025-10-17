/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#ifndef SUDO_TLS_COMMON_H
#define SUDO_TLS_COMMON_H

#include <config.h>

#if defined(HAVE_OPENSSL)
# if defined(HAVE_WOLFSSL)
#  include <wolfssl/options.h>
# endif
# include <openssl/ssl.h>
# include <openssl/err.h>

struct tls_client_closure {
    SSL *ssl;
    void *parent_closure;
    struct sudo_event_base *evbase;	/* duplicated */
    struct sudo_event *tls_connect_ev;
    struct peer_info *peer_name;
    struct timespec connect_timeout;
    bool (*start_fn)(struct tls_client_closure *);
    bool tls_connect_state;
};

/* tls_client.c */
void tls_connect_cb(int sock, int what, void *v);
bool tls_client_setup(int sock, const char *ca_bundle_file, const char *cert_file, const char *key_file, const char *dhparam_file, const char *ciphers_v12, const char *ciphers_v13, bool verify_server, bool check_peer, struct tls_client_closure *closure);
bool tls_ctx_client_setup(SSL_CTX *ssl_ctx, int sock, struct tls_client_closure *closure);

/* tls_init.c */
SSL_CTX *init_tls_context(const char *ca_bundle_file, const char *cert_file, const char *key_file, const char *dhparam_file, const char *ciphers_v12, const char *ciphers_v13, bool verify_cert);

#endif /* HAVE_OPENSSL */

#endif /* SUDO_TLS_COMMON_H */
