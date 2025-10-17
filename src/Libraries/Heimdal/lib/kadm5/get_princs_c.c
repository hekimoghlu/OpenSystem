/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
kadm5_c_get_principals(void *server_handle,
		       const char *expression,
		       char ***princs,
		       int *count)
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
    if (sp == NULL)
	return ENOMEM;
    krb5_store_int32(sp, kadm_get_princs);
    krb5_store_int32(sp, expression != NULL);
    if(expression)
	krb5_store_string(sp, expression);
    ret = _kadm5_client_send(context, sp);
    krb5_storage_free(sp);
    if (ret)
	return ret;
    ret = _kadm5_client_recv(context, &reply);
    if(ret)
	return ret;
    sp = krb5_storage_from_data (&reply);
    if (sp == NULL) {
	krb5_data_free (&reply);
	return ENOMEM;
    }
    krb5_ret_int32(sp, &tmp);
    ret = tmp;
    if(ret == 0) {
	int i;
	krb5_ret_int32(sp, &tmp);
	*princs = calloc(tmp + 1, sizeof(**princs));
	if (*princs == NULL) {
	    ret = ENOMEM;
	    goto out;
	}
	for(i = 0; i < tmp; i++)
	    krb5_ret_string(sp, &(*princs)[i]);
	*count = tmp;
    }
out:
    krb5_storage_free(sp);
    krb5_data_free (&reply);
    return ret;
}
