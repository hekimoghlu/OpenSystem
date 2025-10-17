/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#ifndef tls_var_h
#define tls_var_h

void tls_var_init_lookup_hash(apr_pool_t *pool, apr_hash_t *map);

/**
 * Callback for installation in Apache's 'ssl_var_lookup' hook to provide
 * SSL related variable lookups to other modules.
 */
const char *tls_var_lookup(
    apr_pool_t *p, server_rec *s, conn_rec *c, request_rec *r, const char *name);

/**
 * A connection has been handshaked. Prepare commond TLS variables on this connection.
 */
apr_status_t tls_var_handshake_done(conn_rec *c);

/**
 * A request is ready for processing, add TLS variables r->subprocess_env if applicable.
 * This is a hook function returning OK/DECLINED.
 */
int tls_var_request_fixup(request_rec *r);

#endif /* tls_var_h */