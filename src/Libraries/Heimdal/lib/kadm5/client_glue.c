/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

kadm5_ret_t
kadm5_init_with_password(const char *client_name,
			 const char *password,
			 const char *service_name,
			 kadm5_config_params *realm_params,
			 unsigned long struct_version,
			 unsigned long api_version,
			 void **server_handle)
{
    return kadm5_c_init_with_password(client_name,
				      password,
				      service_name,
				      realm_params,
				      struct_version,
				      api_version,
				      server_handle);
}

kadm5_ret_t
kadm5_init_with_password_ctx(krb5_context context,
			     const char *client_name,
			     const char *password,
			     const char *service_name,
			     kadm5_config_params *realm_params,
			     unsigned long struct_version,
			     unsigned long api_version,
			     void **server_handle)
{
    return kadm5_c_init_with_password_ctx(context,
					  client_name,
					  password,
					  service_name,
					  realm_params,
					  struct_version,
					  api_version,
					  server_handle);
}

kadm5_ret_t
kadm5_init_with_skey(const char *client_name,
		     const char *keytab,
		     const char *service_name,
		     kadm5_config_params *realm_params,
		     unsigned long struct_version,
		     unsigned long api_version,
		     void **server_handle)
{
    return kadm5_c_init_with_skey(client_name,
				  keytab,
				  service_name,
				  realm_params,
				  struct_version,
				  api_version,
				  server_handle);
}

kadm5_ret_t
kadm5_init_with_skey_ctx(krb5_context context,
			 const char *client_name,
			 const char *keytab,
			 const char *service_name,
			 kadm5_config_params *realm_params,
			 unsigned long struct_version,
			 unsigned long api_version,
			 void **server_handle)
{
    return kadm5_c_init_with_skey_ctx(context,
				      client_name,
				      keytab,
				      service_name,
				      realm_params,
				      struct_version,
				      api_version,
				      server_handle);
}

kadm5_ret_t
kadm5_init_with_creds(const char *client_name,
		      krb5_ccache ccache,
		      const char *service_name,
		      kadm5_config_params *realm_params,
		      unsigned long struct_version,
		      unsigned long api_version,
		      void **server_handle)
{
    return kadm5_c_init_with_creds(client_name,
				   ccache,
				   service_name,
				   realm_params,
				   struct_version,
				   api_version,
				   server_handle);
}

kadm5_ret_t
kadm5_init_with_creds_ctx(krb5_context context,
			  const char *client_name,
			  krb5_ccache ccache,
			  const char *service_name,
			  kadm5_config_params *realm_params,
			  unsigned long struct_version,
			  unsigned long api_version,
			  void **server_handle)
{
    return kadm5_c_init_with_creds_ctx(context,
				       client_name,
				       ccache,
				       service_name,
				       realm_params,
				       struct_version,
				       api_version,
				       server_handle);
}
