/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
krb5_mk_priv(krb5_context context,
	     krb5_auth_context auth_context,
	     const krb5_data *userdata,
	     krb5_data *outbuf,
	     krb5_replay_data *outdata)
{
    krb5_error_code ret;
    KRB_PRIV s;
    EncKrbPrivPart part;
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

    memset(&rdata, 0, sizeof(rdata));

    part.user_data = *userdata;

    krb5_us_timeofday (context, &rdata.timestamp, &rdata.usec);

    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_TIME) {
	part.timestamp = &rdata.timestamp;
	part.usec      = &rdata.usec;
    } else {
	part.timestamp = NULL;
	part.usec      = NULL;
    }

    if (auth_context->flags & KRB5_AUTH_CONTEXT_RET_TIME) {
	outdata->timestamp = rdata.timestamp;
	outdata->usec = rdata.usec;
    }

    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE) {
	rdata.seq = auth_context->local_seqnumber;
	part.seq_number = &rdata.seq;
    } else
	part.seq_number = NULL;

    if (auth_context->flags & KRB5_AUTH_CONTEXT_RET_SEQUENCE)
	outdata->seq = auth_context->local_seqnumber;

    part.s_address = auth_context->local_address;
    part.r_address = auth_context->remote_address;

    krb5_data_zero (&s.enc_part.cipher);

    ASN1_MALLOC_ENCODE(EncKrbPrivPart, buf, buf_size, &part, &len, ret);
    if (ret)
	goto fail;
    if (buf_size != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");

    s.pvno = 5;
    s.msg_type = krb_priv;
    s.enc_part.etype = key->keytype;
    s.enc_part.kvno = NULL;

    ret = krb5_crypto_init(context, key, 0, &crypto);
    if (ret) {
	free (buf);
	return ret;
    }
    ret = krb5_encrypt (context,
			crypto,
			KRB5_KU_KRB_PRIV,
			buf + buf_size - len,
			len,
			&s.enc_part.cipher);
    krb5_crypto_destroy(context, crypto);
    if (ret) {
	free(buf);
	return ret;
    }
    free(buf);


    ASN1_MALLOC_ENCODE(KRB_PRIV, buf, buf_size, &s, &len, ret);
    if (ret)
	goto fail;
    if (buf_size != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");

    krb5_data_free (&s.enc_part.cipher);

    ret = krb5_data_copy(outbuf, buf + buf_size - len, len);
    if (ret) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	free(buf);
	return ENOMEM;
    }
    free (buf);
    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE)
	auth_context->local_seqnumber =
	    (auth_context->local_seqnumber + 1) & 0xFFFFFFFF;
    return 0;

  fail:
    free (buf);
    krb5_data_free (&s.enc_part.cipher);
    return ret;
}
