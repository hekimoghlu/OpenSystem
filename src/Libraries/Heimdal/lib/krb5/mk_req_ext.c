/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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

krb5_error_code
_krb5_mk_req_internal(krb5_context context,
		      krb5_auth_context *auth_context,
		      const krb5_flags ap_req_options,
		      krb5_data *in_data,
		      krb5_ccache ccache,
		      krb5_creds *in_creds,
		      krb5_data *outbuf,
		      krb5_key_usage checksum_usage,
		      krb5_key_usage encrypt_usage)
{
    krb5_error_code ret;
    krb5_data authenticator;
    Checksum c;
    Checksum *c_opt;
    krb5_auth_context ac;

    if(auth_context) {
	if(*auth_context == NULL)
	    ret = krb5_auth_con_init(context, auth_context);
	else
	    ret = 0;
	ac = *auth_context;
    } else
	ret = krb5_auth_con_init(context, &ac);
    if(ret)
	return ret;

    if(ac->local_subkey == NULL && (ap_req_options & AP_OPTS_USE_SUBKEY)) {
	ret = krb5_auth_con_generatelocalsubkey(context,
						ac,
						&in_creds->session);
	if(ret)
	    goto out;
    }

    krb5_free_keyblock(context, ac->keyblock);
    ret = krb5_copy_keyblock(context, &in_creds->session, &ac->keyblock);
    if (ret)
	goto out;

    /*
     * Now that we know the enctype, setup PFS if possible
     */
    if (auth_context && (ap_req_options & AP_OPTS_USE_PFS) != 0) {
	ret = _krb5_auth_con_setup_pfs(context, ac, ac->keyblock->keytype);
	if (ret)
	    goto out;
    }

    /* it's unclear what type of checksum we can use.  try the best one, except:
     * a) if it's configured differently for the current realm, or
     * b) if the session key is des-cbc-crc
     */

    if (in_data) {
	if(ac->keyblock->keytype == ETYPE_DES_CBC_CRC) {
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
 */ret = krb5_create_checksum(context,
				       NULL,
				       0,
				       CKSUMTYPE_RSA_MD4,
				       in_data->data,
				       in_data->length,
				       &c);
	} else if(ac->keyblock->keytype == ETYPE_ARCFOUR_HMAC_MD5 ||
		  ac->keyblock->keytype == ETYPE_ARCFOUR_HMAC_MD5_56 ||
		  ac->keyblock->keytype == ETYPE_DES_CBC_MD4 ||
		  ac->keyblock->keytype == ETYPE_DES_CBC_MD5) {
	    /* this is to make MS kdc happy */
	    ret = krb5_create_checksum(context,
				       NULL,
				       0,
				       CKSUMTYPE_RSA_MD5,
				       in_data->data,
				       in_data->length,
				       &c);
	} else {
	    krb5_crypto crypto;

	    ret = krb5_crypto_init(context, ac->keyblock, 0, &crypto);
	    if (ret)
		goto out;
	    ret = krb5_create_checksum(context,
				       crypto,
				       checksum_usage,
				       0,
				       in_data->data,
				       in_data->length,
				       &c);
	    krb5_crypto_destroy(context, crypto);
	}
	c_opt = &c;
    } else {
	c_opt = NULL;
    }

    if (ret)
	goto out;

    ret = _krb5_build_authenticator(context,
				    ac,
				    ac->keyblock->keytype,
				    in_creds,
				    c_opt,
				    &authenticator,
				    encrypt_usage);
    if (c_opt)
	free_Checksum (c_opt);
    if (ret)
	goto out;

    ret = krb5_build_ap_req(context, ac->keyblock->keytype,
			    in_creds, ap_req_options, authenticator, outbuf);
out:
    if(auth_context == NULL)
	krb5_auth_con_free(context, ac);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_mk_req_extended(krb5_context context,
		     krb5_auth_context *auth_context,
		     const krb5_flags ap_req_options,
		     krb5_data *in_data,
		     krb5_creds *in_creds,
		     krb5_data *outbuf)
{
    return _krb5_mk_req_internal(context,
				 auth_context,
				 ap_req_options,
				 in_data,
				 NULL,
				 in_creds,
				 outbuf,
				 KRB5_KU_AP_REQ_AUTH_CKSUM,
				 KRB5_KU_AP_REQ_AUTH);
}

