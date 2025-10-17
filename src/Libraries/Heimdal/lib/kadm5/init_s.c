/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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


static kadm5_ret_t
kadm5_s_init_with_context(krb5_context context,
			  const char *client_name,
			  const char *service_name,
			  kadm5_config_params *realm_params,
			  unsigned long struct_version,
			  unsigned long api_version,
			  void **server_handle)
{
    kadm5_ret_t ret;
    kadm5_server_context *ctx;
    ret = _kadm5_s_init_context(&ctx, realm_params, context);
    if(ret)
	return ret;

    assert(ctx->config.dbname != NULL);
    assert(ctx->config.stash_file != NULL);
    assert(ctx->config.acl_file != NULL);
    assert(ctx->log_context.log_file != NULL);
#ifndef NO_UNIX_SOCKETS
    assert(ctx->log_context.socket_name.sun_path[0] != '\0');
#else
    assert(ctx->log_context.socket_info != NULL);
#endif

    ret = hdb_create(ctx->context, &ctx->db, ctx->config.dbname);
    if(ret)
	return ret;
    ret = hdb_set_master_keyfile (ctx->context,
				  ctx->db, ctx->config.stash_file);
    if(ret)
	return ret;

    ctx->log_context.log_fd   = -1;

#ifndef NO_UNIX_SOCKETS
    ctx->log_context.socket_fd = socket (AF_UNIX, SOCK_DGRAM, 0);
#else
    ctx->log_context.socket_fd = socket (ctx->log_context.socket_info->ai_family,
					 ctx->log_context.socket_info->ai_socktype,
					 ctx->log_context.socket_info->ai_protocol);
#endif
    if (!rk_IS_BAD_SOCKET(ctx->log_context.socket_fd))
	socket_set_nopipe(ctx->log_context.socket_fd, ctx->log_context.socket_fd);

    ret = krb5_parse_name(ctx->context, client_name, &ctx->caller);
    if(ret)
	return ret;

    ret = _kadm5_acl_init(ctx);
    if(ret)
	return ret;

    *server_handle = ctx;
    return 0;
}

kadm5_ret_t
kadm5_s_init_with_password_ctx(krb5_context context,
			       const char *client_name,
			       const char *password,
			       const char *service_name,
			       kadm5_config_params *realm_params,
			       unsigned long struct_version,
			       unsigned long api_version,
			       void **server_handle)
{
    return kadm5_s_init_with_context(context,
				     client_name,
				     service_name,
				     realm_params,
				     struct_version,
				     api_version,
				     server_handle);
}

kadm5_ret_t
kadm5_s_init_with_password(const char *client_name,
			   const char *password,
			   const char *service_name,
			   kadm5_config_params *realm_params,
			   unsigned long struct_version,
			   unsigned long api_version,
			   void **server_handle)
{
    krb5_context context;
    kadm5_ret_t ret;
    kadm5_server_context *ctx;

    ret = krb5_init_context(&context);
    if (ret)
	return ret;
    ret = kadm5_s_init_with_password_ctx(context,
					 client_name,
					 password,
					 service_name,
					 realm_params,
					 struct_version,
					 api_version,
					 server_handle);
    if(ret){
	krb5_free_context(context);
	return ret;
    }
    ctx = *server_handle;
    ctx->my_context = 1;
    return 0;
}

kadm5_ret_t
kadm5_s_init_with_skey_ctx(krb5_context context,
			   const char *client_name,
			   const char *keytab,
			   const char *service_name,
			   kadm5_config_params *realm_params,
			   unsigned long struct_version,
			   unsigned long api_version,
			   void **server_handle)
{
    return kadm5_s_init_with_context(context,
				     client_name,
				     service_name,
				     realm_params,
				     struct_version,
				     api_version,
				     server_handle);
}

kadm5_ret_t
kadm5_s_init_with_skey(const char *client_name,
		       const char *keytab,
		       const char *service_name,
		       kadm5_config_params *realm_params,
		       unsigned long struct_version,
		       unsigned long api_version,
		       void **server_handle)
{
    krb5_context context;
    kadm5_ret_t ret;
    kadm5_server_context *ctx;

    ret = krb5_init_context(&context);
    if (ret)
	return ret;
    ret = kadm5_s_init_with_skey_ctx(context,
				     client_name,
				     keytab,
				     service_name,
				     realm_params,
				     struct_version,
				     api_version,
				     server_handle);
    if(ret){
	krb5_free_context(context);
	return ret;
    }
    ctx = *server_handle;
    ctx->my_context = 1;
    return 0;
}

kadm5_ret_t
kadm5_s_init_with_creds_ctx(krb5_context context,
			    const char *client_name,
			    krb5_ccache ccache,
			    const char *service_name,
			    kadm5_config_params *realm_params,
			    unsigned long struct_version,
			    unsigned long api_version,
			    void **server_handle)
{
    return kadm5_s_init_with_context(context,
				     client_name,
				     service_name,
				     realm_params,
				     struct_version,
				     api_version,
				     server_handle);
}

kadm5_ret_t
kadm5_s_init_with_creds(const char *client_name,
			krb5_ccache ccache,
			const char *service_name,
			kadm5_config_params *realm_params,
			unsigned long struct_version,
			unsigned long api_version,
			void **server_handle)
{
    krb5_context context;
    kadm5_ret_t ret;
    kadm5_server_context *ctx;

    ret = krb5_init_context(&context);
    if (ret)
	return ret;
    ret = kadm5_s_init_with_creds_ctx(context,
				      client_name,
				      ccache,
				      service_name,
				      realm_params,
				      struct_version,
				      api_version,
				      server_handle);
    if(ret){
	krb5_free_context(context);
	return ret;
    }
    ctx = *server_handle;
    ctx->my_context = 1;
    return 0;
}
