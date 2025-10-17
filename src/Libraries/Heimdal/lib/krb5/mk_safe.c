/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#include "krb5_locl.h"

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_mk_safe(krb5_context context,
	     krb5_auth_context auth_context,
	     const krb5_data *userdata,
	     krb5_data *outbuf,
	     krb5_replay_data *outdata)
{
    krb5_error_code ret;
    KRB_SAFE s;
    u_char *buf = NULL;
    size_t buf_size;
    size_t len = 0;
    krb5_crypto crypto;
    krb5_keyblock *key;
    krb5_replay_data rdata;

    if ((auth_context->flags &
	 (KRB5_AUTH_CONTEXT_RET_TIME | KRB5_AUTH_CONTEXT_RET_SEQUENCE)) &&
	outdata == NULL)
	return KRB5_RC_REQUIRED; /* XXX better error, MIT returns this */

    if (auth_context->local_subkey)
	key = auth_context->local_subkey;
    else if (auth_context->remote_subkey)
	key = auth_context->remote_subkey;
    else
	key = auth_context->keyblock;

    s.pvno = 5;
    s.msg_type = krb_safe;

    memset(&rdata, 0, sizeof(rdata));

    s.safe_body.user_data = *userdata;

    krb5_us_timeofday (context, &rdata.timestamp, &rdata.usec);

    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_TIME) {
	s.safe_body.timestamp  = &rdata.timestamp;
	s.safe_body.usec       = &rdata.usec;
    } else {
	s.safe_body.timestamp  = NULL;
	s.safe_body.usec       = NULL;
    }

    if (auth_context->flags & KRB5_AUTH_CONTEXT_RET_TIME) {
	outdata->timestamp = rdata.timestamp;
	outdata->usec = rdata.usec;
    }

    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE) {
	rdata.seq = auth_context->local_seqnumber;
	s.safe_body.seq_number = &rdata.seq;
    } else
	s.safe_body.seq_number = NULL;

    if (auth_context->flags & KRB5_AUTH_CONTEXT_RET_SEQUENCE)
	outdata->seq = auth_context->local_seqnumber;

    s.safe_body.s_address = auth_context->local_address;
    s.safe_body.r_address = auth_context->remote_address;

    s.cksum.cksumtype       = 0;
    s.cksum.checksum.data   = NULL;
    s.cksum.checksum.length = 0;

    ASN1_MALLOC_ENCODE(KRB_SAFE, buf, buf_size, &s, &len, ret);
    if (ret)
	return ret;
    if(buf_size != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");
    ret = krb5_crypto_init(context, key, 0, &crypto);
    if (ret) {
	free (buf);
	return ret;
    }
    ret = krb5_create_checksum(context,
			       crypto,
			       KRB5_KU_KRB_SAFE_CKSUM,
			       0,
			       buf,
			       len,
			       &s.cksum);
    krb5_crypto_destroy(context, crypto);
    if (ret) {
	free (buf);
	return ret;
    }

    free(buf);
    ASN1_MALLOC_ENCODE(KRB_SAFE, buf, buf_size, &s, &len, ret);
    free_Checksum (&s.cksum);
    if(ret)
	return ret;
    if(buf_size != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");

    outbuf->length = len;
    outbuf->data   = buf;
    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE)
	auth_context->local_seqnumber =
	    (auth_context->local_seqnumber + 1) & 0xFFFFFFFF;
    return 0;
}
