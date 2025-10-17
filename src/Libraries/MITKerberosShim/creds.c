/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#include "heim.h"
#include <string.h>
#include <errno.h>
#include <syslog.h>


mit_krb5_error_code KRB5_CALLCONV
krb5_decode_ticket(const mit_krb5_data *code, 
		   mit_krb5_ticket **rep)
{
    krb5_error_code ret;
    Ticket t;
    
    LOG_ENTRY();

    ret = decode_Ticket((unsigned char *)code->data, code->length, &t, NULL);
    if (ret)
	return ret;
    
    *rep = calloc(1, sizeof(**rep));

    /* XXX */
    (*rep)->enc_part.kvno = t.enc_part.kvno ? *t.enc_part.kvno : 0;

    free_Ticket(&t);

    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_get_credentials(mit_krb5_context context,
		     mit_krb5_flags flags,
		     mit_krb5_ccache id,
		     mit_krb5_creds *mcreds,
		     mit_krb5_creds **creds)
{
    krb5_error_code ret;
    krb5_flags options = flags;
    krb5_creds *hcreds = NULL, hmcreds;

    LOG_ENTRY();

    mshim_mcred2hcred(HC(context), mcreds, &hmcreds);

    ret = heim_krb5_get_credentials(HC(context), options, (krb5_ccache)id, &hmcreds, &hcreds);

    heim_krb5_free_cred_contents(HC(context), &hmcreds);
    if (ret == 0) {
	*creds = calloc(1, sizeof(**creds));
	mshim_hcred2mcred(HC(context), hcreds, *creds);
	heim_krb5_free_creds(HC(context), hcreds);
    }

    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_copy_creds(mit_krb5_context context,
		const mit_krb5_creds *from,
		mit_krb5_creds **to)
{
    mit_krb5_error_code ret;
    mit_krb5_creds *c;

    c = mshim_malloc(sizeof(*c));

    c->magic = MIT_KV5M_CREDS;

    ret = krb5_copy_principal(context, from->client, &c->client);
    if (ret)
	abort();
    ret = krb5_copy_principal(context, from->server, &c->server);
    if (ret)
	abort();
    
    ret = krb5_copy_keyblock_contents(context, &from->keyblock,
				      &c->keyblock);
    if (ret)
	abort();

    c->ticket.magic = MIT_KV5M_DATA;
    c->ticket.length = from->ticket.length;
    c->ticket.data = mshim_malloc(from->ticket.length);
    memcpy(c->ticket.data, from->ticket.data, c->ticket.length);

    c->times.authtime = from->times.authtime;
    c->times.starttime = from->times.starttime;
    c->times.endtime = from->times.endtime;
    c->times.renew_till = from->times.renew_till;

    c->ticket_flags = from->ticket_flags;

    *to = c;

    return 0;
}
