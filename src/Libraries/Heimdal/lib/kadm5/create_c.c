/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
kadm5_c_create_principal(void *server_handle,
			 kadm5_principal_ent_t princ,
			 uint32_t mask,
			 const char *password,
			 int n_ks_tuple,
			 krb5_key_salt_tuple *ks_tuple)
{
    kadm5_client_context *context = server_handle;
    kadm5_ret_t ret;
    krb5_storage *sp;
    unsigned char buf[1024];
    int32_t tmp;
    krb5_data reply;

    ret = _kadm5_connect(server_handle);
    if(ret)
	return ret;

    sp = krb5_storage_from_mem(buf, sizeof(buf));
    if (sp == NULL) {
	krb5_clear_error_message(context->context);
	return ENOMEM;
    }
    krb5_store_int32(sp, kadm_create);
    kadm5_store_principal_ent(sp, princ);
    krb5_store_int32(sp, mask);
    krb5_store_string(sp, password);
    _kadm5_store_ks_tuple(sp, n_ks_tuple, ks_tuple);
    ret = _kadm5_client_send(context, sp);
    krb5_storage_free(sp);
    if (ret)
	return ret;
    ret = _kadm5_client_recv(context, &reply);
    if(ret)
	return ret;
    sp = krb5_storage_from_data (&reply);
    if (sp == NULL) {
	krb5_clear_error_message(context->context);
	krb5_data_free (&reply);
	return ENOMEM;
    }
    krb5_ret_int32(sp, &tmp);
    krb5_clear_error_message(context->context);
    krb5_storage_free(sp);
    krb5_data_free (&reply);
    return tmp;
}

