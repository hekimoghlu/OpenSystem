/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include "kdc_locl.h"

static krb5plugin_windc_ftable *windcft;
static void *windcctx;

/*
 * Pick the first WINDC module that we find.
 */

krb5_error_code
krb5_kdc_windc_init(krb5_context context)
{
    struct krb5_plugin *list = NULL, *e;
    krb5_error_code ret;

    ret = _krb5_plugin_find(context, PLUGIN_TYPE_DATA, "windc", &list);
    if(ret != 0 || list == NULL)
	return 0;

    for (e = list; e != NULL; e = _krb5_plugin_get_next(e)) {

	windcft = _krb5_plugin_get_symbol(e);
	if (windcft->minor_version < KRB5_WINDC_PLUGIN_MINOR)
	    continue;

	(*windcft->init)(context, &windcctx);
	break;
    }
    _krb5_plugin_free(list);
    if (e == NULL) {
	krb5_set_error_message(context, ENOENT, "Did not find any WINDC plugin");
	windcft = NULL;
	return ENOENT;
    }

    return 0;
}


krb5_error_code
_kdc_pac_generate(krb5_context context,
		  hdb_entry_ex *client,
		  krb5_pac *pac)
{
    *pac = NULL;
    if (windcft == NULL)
	return 0;
    return (windcft->pac_generate)(windcctx, context, client, pac);
}

krb5_error_code
_kdc_pac_verify(krb5_context context,
		const krb5_principal client_principal,
		const krb5_principal delegated_proxy_principal,
		hdb_entry_ex *client,
		hdb_entry_ex *server,
		hdb_entry_ex *krbtgt,
		krb5_pac *pac,
		int *verified)
{
    krb5_error_code ret;

    if (windcft == NULL)
	return 0;

    ret = windcft->pac_verify(windcctx, context,
			      client_principal,
			      delegated_proxy_principal,
			      client, server, krbtgt, pac);
    if (ret == 0)
	*verified = 1;
    return ret;
}

krb5_error_code
_kdc_check_access(krb5_context context,
		  krb5_kdc_configuration *config,
		  hdb_entry_ex *client_ex, const char *client_name,
		  hdb_entry_ex *server_ex, const char *server_name,
		  KDC_REQ *req,
		  METHOD_DATA *method_data)
{
    if (windcft == NULL || windcft->client_access == NULL)
	    return kdc_check_flags(context, config,
				   client_ex, client_name,
				   server_ex, server_name,
				   req->msg_type == krb_as_req);

    return (windcft->client_access)(windcctx, 
				    context, config, 
				    client_ex, client_name, 
				    server_ex, server_name, 
				    req, method_data);
}
