/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
krb5_mk_rep(krb5_context context,
	    krb5_auth_context auth_context,
	    krb5_data *outbuf)
{
    krb5_crypto crypto = NULL;
    krb5_error_code ret;
    uint8_t *buf = NULL;
    EncAPRepPart body;
    size_t buf_size;
    size_t len = 0;
    AP_REP ap;

    memset (&ap, 0, sizeof(ap));
    memset (&body, 0, sizeof(body));

    ap.pvno = 5;
    ap.msg_type = krb_ap_rep;

    body.ctime = auth_context->authenticator->ctime;
    body.cusec = auth_context->authenticator->cusec;
    if (auth_context->flags & KRB5_AUTH_CONTEXT_USE_SUBKEY) {
	if (auth_context->local_subkey == NULL) {
	    ret = krb5_auth_con_generatelocalsubkey(context,
						    auth_context,
						    auth_context->keyblock);
	    if(ret)
		goto out;
	}
	ret = krb5_copy_keyblock(context, auth_context->local_subkey,
				 &body.subkey);
	if (ret) {
	    ret = krb5_enomem(context);
	    goto out;
	}
    } else
	body.subkey = NULL;
    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE) {
	if(auth_context->local_seqnumber == 0)
	    krb5_generate_seq_number (context,
				      auth_context->keyblock,
				      &auth_context->local_seqnumber);
	ALLOC(body.seq_number, 1);
	if (body.seq_number == NULL) {
	    ret = krb5_enomem(context);
	    goto out;
	}
	*(body.seq_number) = auth_context->local_seqnumber;
    } else
	body.seq_number = NULL;

    ap.enc_part.etype = auth_context->keyblock->keytype;
    ap.enc_part.kvno  = NULL;

    /*
     * Use CF2 to merge DH key all 3 keys, session and both subkeys keys.
     */

    if (auth_context->pfs) {
	ret = _krb5_pfs_mk_rep(context, auth_context, &ap);
	if (ret)
	    goto out;
    }

    /*
     *
     */

    ASN1_MALLOC_ENCODE(EncAPRepPart, buf, buf_size, &body, &len, ret);
    if(ret)
	goto out;
    if (buf_size != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");

    ret = krb5_crypto_init(context, auth_context->keyblock,
			   0 /* ap.enc_part.etype */, &crypto);
    if (ret)
	goto out;

    ret = krb5_encrypt (context,
			crypto,
			KRB5_KU_AP_REQ_ENC_PART,
			buf + buf_size - len,
			len,
			&ap.enc_part.cipher);
    if (ret)
	goto out;

    ASN1_MALLOC_ENCODE(AP_REP, outbuf->data, outbuf->length, &ap, &len, ret);
    if (ret == 0 && outbuf->length != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");
 out:
    if (buf)
	free(buf);
    if (crypto)
	krb5_crypto_destroy(context, crypto);
    free_EncAPRepPart (&body);
    free_AP_REP (&ap);

    return ret;
}
