/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#define HC_DEPRECATED_CRYPTO

#include "mech_locl.h"

#include <krb5.h>
#include <roken.h>

#include "crypto-headers.h"


OM_uint32
gss_mg_gen_cb(OM_uint32 *__nonnull minor_status,
	      __nonnull const gss_channel_bindings_t b,
	      uint8_t p[16],
	      __nullable gss_buffer_t buffer)
{
    krb5_error_code ret;
    krb5_ssize_t sret;
    krb5_storage *sp;
    krb5_data data;

    krb5_data_zero(&data);

    sp = krb5_storage_emem();
    if (sp == NULL) {
	*minor_status = ENOMEM;
	goto out;
    }
    krb5_storage_set_byteorder(sp, KRB5_STORAGE_BYTEORDER_LE);

    ret = krb5_store_uint32(sp, b->initiator_addrtype);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    ret = krb5_store_uint32(sp, (uint32_t)b->initiator_address.length);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    sret = krb5_storage_write(sp, b->initiator_address.value,
			     (uint32_t)b->initiator_address.length);
    if (sret < 0 || (size_t)sret != b->initiator_address.length) {
	*minor_status = ENOMEM;
	goto out;
    }
	
    ret = krb5_store_uint32(sp, b->acceptor_addrtype);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    ret = krb5_store_uint32(sp, (uint32_t)b->acceptor_address.length);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    sret = krb5_storage_write(sp, b->acceptor_address.value,
			     b->acceptor_address.length);
    if (sret < 0 || (size_t)sret != b->acceptor_address.length) {
	*minor_status = ENOMEM;
	goto out;
    }

    ret = krb5_store_uint32(sp, (uint32_t)b->application_data.length);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    sret = krb5_storage_write(sp, b->application_data.value,
			      b->application_data.length);
    if (sret < 0 || (size_t)sret != b->application_data.length) {
	*minor_status = ENOMEM;
	goto out;
    }

    ret = krb5_storage_to_data(sp, &data);
    if (ret) {
	*minor_status = ret;
	goto out;
    }
    
    CCDigest(kCCDigestMD5, data.data, data.length, p);

    if (buffer) {
	buffer->value = data.data;
	buffer->length = data.length;
    } else {
	krb5_data_free(&data);
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;

 out:
    if (sp)
	krb5_storage_free(sp);
    return GSS_S_FAILURE;
}

OM_uint32
gss_mg_validate_cb(OM_uint32 *__nonnull minor_status,
		   __nonnull const gss_channel_bindings_t b,
		   const uint8_t p[16],
		   __nonnull gss_buffer_t buffer)
{
    static uint8_t zeros[16] = { 0 };
    OM_uint32 major_status, junk;
    uint8_t hash[16];

    if (b != GSS_C_NO_CHANNEL_BINDINGS
	&& memcmp(p, zeros, sizeof(zeros)) != 0) {

	major_status = gss_mg_gen_cb(minor_status, b, hash, buffer);
	if (major_status)
	    return major_status;

	if(ct_memcmp(hash, p, sizeof(hash)) != 0) {
	    gss_release_buffer(&junk, buffer);
	    *minor_status = 0;
	    return GSS_S_BAD_BINDINGS;
	}
    } else {
	buffer->length = 0;
	buffer->value = NULL;
    }

    return GSS_S_COMPLETE;
}
