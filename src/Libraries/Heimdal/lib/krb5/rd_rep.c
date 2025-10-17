/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
krb5_rd_rep(krb5_context context,
	    krb5_auth_context auth_context,
	    const krb5_data *inbuf,
	    krb5_ap_rep_enc_part **repl)
{
    krb5_error_code ret;
    AP_REP ap_rep;
    size_t len;
    krb5_data data;
    krb5_crypto crypto;

    krb5_data_zero (&data);

    ret = decode_AP_REP(inbuf->data, inbuf->length, &ap_rep, &len);
    if (ret)
	return ret;
    if (ap_rep.pvno != 5) {
	ret = KRB5KRB_AP_ERR_BADVERSION;
	krb5_clear_error_message (context);
	goto out;
    }
    if (ap_rep.msg_type != krb_ap_rep) {
	ret = KRB5KRB_AP_ERR_MSG_TYPE;
	krb5_clear_error_message (context);
	goto out;
    }

    if (ap_rep.pfs && auth_context->pfs) {
	ret = _krb5_pfs_rd_rep(context, auth_context, &ap_rep);
	if (ret)
	    goto out;
    } else {
	_krb5_auth_con_free_pfs(context, auth_context);
    }

    ret = krb5_crypto_init(context, auth_context->keyblock, 0, &crypto);
    if (ret)
	goto out;
    ret = krb5_decrypt_EncryptedData (context,
				      crypto,
				      KRB5_KU_AP_REQ_ENC_PART,
				      &ap_rep.enc_part,
				      &data);
    krb5_crypto_destroy(context, crypto);
    if (ret)
	goto out;

    *repl = malloc(sizeof(**repl));
    if (*repl == NULL) {
	ret = ENOMEM;
	krb5_set_error_message(context, ret, N_("malloc: out of memory", ""));
	goto out;
    }
    ret = decode_EncAPRepPart(data.data, data.length, *repl, &len);
    if (ret) {
	krb5_set_error_message(context, ret, N_("Failed to decode EncAPRepPart", ""));
	return ret;
    }

    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_TIME) {
	if ((*repl)->ctime != auth_context->authenticator->ctime ||
	    (*repl)->cusec != auth_context->authenticator->cusec)
	{
	    krb5_free_ap_rep_enc_part(context, *repl);
	    *repl = NULL;
	    ret = KRB5KRB_AP_ERR_MUT_FAIL;
	    krb5_clear_error_message (context);
	    goto out;
	}
    }
    if ((*repl)->seq_number)
	krb5_auth_con_setremoteseqnumber(context, auth_context,
					 *((*repl)->seq_number));
    if ((*repl)->subkey) {

	if (ap_rep.pfs && auth_context->pfs) {
	    ret = _krb5_pfs_update_key(context, auth_context,
				       "server key",
				       (*repl)->subkey);
	    if (ret)
		goto out;
	} else {
	    _krb5_debugx(context, 10, "krb5_rd_rep not using PFS");
	}

	krb5_auth_con_setremotesubkey(context, auth_context, (*repl)->subkey);
    }

 out:
    _krb5_auth_con_free_pfs(context, auth_context);
    krb5_data_free (&data);
    free_AP_REP (&ap_rep);
    return ret;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_ap_rep_enc_part (krb5_context context,
			   krb5_ap_rep_enc_part *val)
{
    if (val) {
	free_EncAPRepPart (val);
	free (val);
    }
}
