/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
krb5_mk_error(krb5_context context,
	      krb5_error_code error_code,
	      const char *e_text,
	      const krb5_data *e_data,
	      const krb5_principal client,
	      const krb5_principal server,
	      time_t *client_time,
	      int *client_usec,
	      krb5_data *reply)
{
    const char *e_text2 = NULL;
    KRB_ERROR msg;
    krb5_timestamp sec;
    int32_t usec;
    size_t len = 0;
    krb5_error_code ret = 0;

    krb5_us_timeofday (context, &sec, &usec);

    memset(&msg, 0, sizeof(msg));
    msg.pvno     = 5;
    msg.msg_type = krb_error;
    msg.stime    = sec;
    msg.susec    = usec;
    msg.ctime    = client_time;
    msg.cusec    = client_usec;
    /* Make sure we only send `protocol' error codes */
    if(error_code < KRB5KDC_ERR_NONE || error_code >= KRB5_ERR_RCSID) {
	if(e_text == NULL)
	    e_text = e_text2 = krb5_get_error_message(context, error_code);
	error_code = KRB5KRB_ERR_GENERIC;
    }
    msg.error_code = error_code - KRB5KDC_ERR_NONE;
    if (e_text)
	msg.e_text = rk_UNCONST(&e_text);
    if (e_data)
	msg.e_data = rk_UNCONST(e_data);
    if(server){
	msg.realm = server->realm;
	msg.sname = server->name;
    }else{
	static char unspec[] = "<unspecified realm>";
	msg.realm = unspec;
    }
    if(client){
	msg.crealm = &client->realm;
	msg.cname = &client->name;
    }

    ASN1_MALLOC_ENCODE(KRB_ERROR, reply->data, reply->length, &msg, &len, ret);
    if (e_text2)
	krb5_free_error_message(context, e_text2);
    if (ret)
	return ret;
    if(reply->length != len)
	krb5_abortx(context, "internal error in ASN.1 encoder");
    return 0;
}
