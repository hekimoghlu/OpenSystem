/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
krb5_rd_priv(krb5_context context,
	     krb5_auth_context auth_context,
	     const krb5_data *inbuf,
	     krb5_data *outbuf,
	     krb5_replay_data *outdata)
{
    krb5_error_code ret;
    KRB_PRIV priv;
    EncKrbPrivPart part;
    size_t len;
    krb5_data plain;
    krb5_keyblock *key;
    krb5_crypto crypto;

    krb5_data_zero(outbuf);

    if ((auth_context->flags &
	 (KRB5_AUTH_CONTEXT_RET_TIME | KRB5_AUTH_CONTEXT_RET_SEQUENCE)))
    {
	if (outdata == NULL) {
	    krb5_clear_error_message (context);
	    return KRB5_RC_REQUIRED; /* XXX better error, MIT returns this */
	}
	/* if these fields are not present in the priv-part, silently
           return zero */
	memset(outdata, 0, sizeof(*outdata));
    }

    memset(&priv, 0, sizeof(priv));
    ret = decode_KRB_PRIV (inbuf->data, inbuf->length, &priv, &len);
    if (ret) {
	krb5_clear_error_message (context);
	goto failure;
    }
    if (priv.pvno != 5) {
	krb5_clear_error_message (context);
	ret = KRB5KRB_AP_ERR_BADVERSION;
	goto failure;
    }
    if (priv.msg_type != krb_priv) {
	krb5_clear_error_message (context);
	ret = KRB5KRB_AP_ERR_MSG_TYPE;
	goto failure;
    }

    if (auth_context->remote_subkey)
	key = auth_context->remote_subkey;
    else if (auth_context->local_subkey)
	key = auth_context->local_subkey;
    else
	key = auth_context->keyblock;

    ret = krb5_crypto_init(context, key, 0, &crypto);
    if (ret)
	goto failure;
    ret = krb5_decrypt_EncryptedData(context,
				     crypto,
				     KRB5_KU_KRB_PRIV,
				     &priv.enc_part,
				     &plain);
    krb5_crypto_destroy(context, crypto);
    if (ret)
	goto failure;

    ret = decode_EncKrbPrivPart (plain.data, plain.length, &part, &len);
    krb5_data_free (&plain);
    if (ret) {
	krb5_clear_error_message (context);
	goto failure;
    }

    /* check sender address */

    if (part.s_address
	&& auth_context->remote_address
	&& !krb5_address_compare (context,
				  auth_context->remote_address,
				  part.s_address)) {
	krb5_clear_error_message (context);
	ret = KRB5KRB_AP_ERR_BADADDR;
	goto failure_part;
    }

    /* check receiver address */

    if (part.r_address
	&& auth_context->local_address
	&& !krb5_address_compare (context,
				  auth_context->local_address,
				  part.r_address)) {
	krb5_clear_error_message (context);
	ret = KRB5KRB_AP_ERR_BADADDR;
	goto failure_part;
    }

    /* check timestamp */
    if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_TIME) {
	krb5_timestamp sec;

	krb5_timeofday (context, &sec);
	if (part.timestamp == NULL ||
	    part.usec      == NULL ||
	    krb5_time_abs(*part.timestamp, sec) > context->max_skew) {
	    krb5_clear_error_message (context);
	    ret = KRB5KRB_AP_ERR_SKEW;
	    goto failure_part;
	}
    }

    /* XXX - check replay cache */
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
 */if (auth_context->flags & KRB5_AUTH_CONTEXT_DO_SEQUENCE) {
	if ((part.seq_number == NULL
	     && auth_context->remote_seqnumber != 0)
	    || (part.seq_number != NULL
		&& *part.seq_number != auth_context->remote_seqnumber)) {
	    krb5_clear_error_message (context);
	    ret = KRB5KRB_AP_ERR_BADORDER;
	    goto failure_part;
	}
	auth_context->remote_seqnumber++;
    }

    ret = krb5_data_copy (outbuf, part.user_data.data, part.user_data.length);
    if (ret)
	goto failure_part;

    if ((auth_context->flags &
	 (KRB5_AUTH_CONTEXT_RET_TIME | KRB5_AUTH_CONTEXT_RET_SEQUENCE))) {
	if(part.timestamp)
	    outdata->timestamp = *part.timestamp;
	if(part.usec)
	    outdata->usec = *part.usec;
	if(part.seq_number)
	    outdata->seq = *part.seq_number;
    }

  failure_part:
    free_EncKrbPrivPart (&part);

  failure:
    free_KRB_PRIV (&priv);
    return ret;
}

