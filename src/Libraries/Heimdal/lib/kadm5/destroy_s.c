/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#include "kadm5_locl.h"

RCSID("$Id$");

/*
 * dealloc a `kadm5_config_params'
 */

static void
destroy_config (kadm5_config_params *c)
{
    free (c->realm);
    free (c->dbname);
    free (c->acl_file);
    free (c->stash_file);
}

/*
 * dealloc a kadm5_log_context
 */

static void
destroy_kadm5_log_context (kadm5_log_context *c)
{
    free (c->log_file);
    rk_closesocket (c->socket_fd);
#ifdef NO_UNIX_SOCKETS
    if (c->socket_info) {
	freeaddrinfo(c->socket_info);
	c->socket_info = NULL;
    }
#endif
}

/*
 * destroy a kadm5 handle
 */

kadm5_ret_t
kadm5_s_destroy(void *server_handle)
{
    kadm5_ret_t ret;
    kadm5_server_context *context = server_handle;
    krb5_context kcontext = context->context;

    ret = context->db->hdb_destroy(kcontext, context->db);
    destroy_kadm5_log_context (&context->log_context);
    destroy_config (&context->config);
    krb5_free_principal (kcontext, context->caller);
    if(context->my_context)
	krb5_free_context(kcontext);
    free (context);
    return ret;
}
