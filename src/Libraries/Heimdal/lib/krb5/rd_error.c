/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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
krb5_rd_error(krb5_context context,
	      const krb5_data *msg,
	      KRB_ERROR *result)
{

    size_t len;
    krb5_error_code ret;

    ret = decode_KRB_ERROR(msg->data, msg->length, result, &len);
    if(ret) {
	krb5_clear_error_message(context);
	return ret;
    }
    result->error_code += KRB5KDC_ERR_NONE;
    return 0;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_error_contents (krb5_context context,
			  krb5_error *error)
{
    free_KRB_ERROR(error);
    memset(error, 0, sizeof(*error));
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_error (krb5_context context,
		 krb5_error *error)
{
    krb5_free_error_contents (context, error);
    free (error);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_error_from_rd_error(krb5_context context,
			 const krb5_error *error,
			 const krb5_creds *creds)
{
    krb5_error_code ret;

    ret = error->error_code;
    if (error->e_text != NULL) {
	krb5_set_error_message(context, ret, N_("Error from KDC: %s", ""),
			       *error->e_text);
    } else {
	char clientname[256], servername[256];

	if (creds != NULL) {
	    krb5_unparse_name_fixed(context, creds->client,
				    clientname, sizeof(clientname));
	    krb5_unparse_name_fixed(context, creds->server,
				    servername, sizeof(servername));
	}

	switch (ret) {
	case KRB5KDC_ERR_NAME_EXP :
	    krb5_set_error_message(context, ret,
				   N_("Client %s%s%s expired", ""),
				   creds ? "(" : "",
				   creds ? clientname : "",
				   creds ? ")" : "");
	    break;
	case KRB5KDC_ERR_SERVICE_EXP :
	    krb5_set_error_message(context, ret,
				   N_("Server %s%s%s expired", ""),
				   creds ? "(" : "",
				   creds ? servername : "",
				   creds ? ")" : "");
	    break;
	case KRB5KDC_ERR_C_PRINCIPAL_UNKNOWN :
	    krb5_set_error_message(context, ret,
				   N_("Client %s%s%s unknown", ""),
				   creds ? "(" : "",
				   creds ? clientname : "",
				   creds ? ")" : "");
	    break;
	case KRB5KDC_ERR_S_PRINCIPAL_UNKNOWN :
	    krb5_set_error_message(context, ret,
				   N_("Server %s%s%s unknown", ""),
				   creds ? "(" : "",
				   creds ? servername : "",
				   creds ? ")" : "");
	    break;
	default :
	    krb5_clear_error_message(context);
	    break;
	}
    }
    return ret;
}
