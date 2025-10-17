/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
_kadm5_client_send(kadm5_client_context *context, krb5_storage *sp)
{
    krb5_data msg, out;
    krb5_error_code ret;
    size_t len;
    krb5_storage *sock;

    assert(context->sock != -1);

    len = krb5_storage_seek(sp, 0, SEEK_CUR);
    ret = krb5_data_alloc(&msg, len);
    if (ret) {
	krb5_clear_error_message(context->context);
	return ret;
    }
    krb5_storage_seek(sp, 0, SEEK_SET);
    krb5_storage_read(sp, msg.data, msg.length);

    ret = krb5_mk_priv(context->context, context->ac, &msg, &out, NULL);
    krb5_data_free(&msg);
    if(ret)
	return ret;

    sock = krb5_storage_from_fd(context->sock);
    if(sock == NULL) {
	krb5_clear_error_message(context->context);
	krb5_data_free(&out);
	return ENOMEM;
    }

    ret = krb5_store_data(sock, out);
    if (ret)
	krb5_clear_error_message(context->context);
    krb5_storage_free(sock);
    krb5_data_free(&out);
    return ret;
}

kadm5_ret_t
_kadm5_client_recv(kadm5_client_context *context, krb5_data *reply)
{
    krb5_error_code ret;
    krb5_data data;
    krb5_storage *sock;

    sock = krb5_storage_from_fd(context->sock);
    if(sock == NULL) {
	krb5_clear_error_message(context->context);
	return ENOMEM;
    }
    ret = krb5_ret_data(sock, &data);
    krb5_storage_free(sock);
    krb5_clear_error_message(context->context);
    if(ret == KRB5_CC_END)
	return KADM5_RPC_ERROR;
    else if(ret)
	return ret;

    ret = krb5_rd_priv(context->context, context->ac, &data, reply, NULL);
    krb5_data_free(&data);
    return ret;
}

krb5_error_code
_kadm5_store_ks_tuple(krb5_storage *sp, uint32_t n_ks_tuple, krb5_key_salt_tuple *ks_tuple)
{
    krb5_error_code ret;
    size_t n;

    ret = krb5_store_uint32(sp, n_ks_tuple);
    if (ret)
	return ret;

    for (n = 0; n < n_ks_tuple; n++) {
	ret = krb5_store_int32(sp, 8); /* element size */
	if (ret) return ret;
	ret = krb5_store_int32(sp, ks_tuple[n].ks_enctype);
	if (ret) return ret;
	ret = krb5_store_int32(sp, ks_tuple[n].ks_salttype);
	if (ret) return ret;
    }
    return 0;
}

krb5_error_code
_kadm5_ret_ks_tuple(krb5_storage *sp, uint32_t *n_ks_tuple, krb5_key_salt_tuple **ks_tuple)
{
    krb5_error_code ret;
    uint32_t elsize, n;

    *n_ks_tuple = 0;
    *ks_tuple = NULL;

    ret = krb5_ret_uint32(sp, n_ks_tuple);
    if (ret == HEIM_ERR_EOF) {
	return 0;
    } else if (ret)
	return ret;

    if (n_ks_tuple < 0 || *n_ks_tuple > 1000) {
	ret = EOVERFLOW;
	goto fail;
    }

    ks_tuple = calloc(*n_ks_tuple, sizeof (*ks_tuple));
    if (ks_tuple == NULL) {
	ret = errno;
	goto fail;
    }
    
    for (n = 0; n < *n_ks_tuple; n++) {
	ret = krb5_ret_uint32(sp, &elsize);
	if (ret)
	    goto fail;
	if (elsize < 8) {
	    ret = EOVERFLOW;
	    goto fail;
	}

	ret = krb5_ret_int32(sp, &(*ks_tuple)[n].ks_enctype);
	if (ret != 0)
	    goto fail;

	ret = krb5_ret_int32(sp, &(*ks_tuple)[n].ks_salttype);
	if (ret != 0)
	    goto fail;

	krb5_storage_seek(sp, elsize - 8, SEEK_CUR);
    }

    return 0;

 fail:
    free(*ks_tuple);
    *ks_tuple = NULL;
    *n_ks_tuple = 0;

    return ret;
}
