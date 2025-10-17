/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
krb5_mk_req_exact(krb5_context context,
		  krb5_auth_context *auth_context,
		  const krb5_flags ap_req_options,
		  const krb5_principal server,
		  krb5_data *in_data,
		  krb5_ccache ccache,
		  krb5_data *outbuf)
{
    krb5_error_code ret;
    krb5_creds this_cred, *cred;

    memset(&this_cred, 0, sizeof(this_cred));

    ret = krb5_cc_get_principal(context, ccache, &this_cred.client);

    if(ret)
	return ret;

    ret = krb5_copy_principal (context, server, &this_cred.server);
    if (ret) {
	krb5_free_cred_contents (context, &this_cred);
	return ret;
    }

    this_cred.times.endtime = 0;
    if (auth_context && *auth_context && (*auth_context)->keytype)
	this_cred.session.keytype = (*auth_context)->keytype;

    ret = krb5_get_credentials (context, 0, ccache, &this_cred, &cred);
    krb5_free_cred_contents(context, &this_cred);
    if (ret)
	return ret;

    ret = krb5_mk_req_extended (context,
				auth_context,
				ap_req_options,
				in_data,
				cred,
				outbuf);
    krb5_free_creds(context, cred);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_mk_req(krb5_context context,
	    krb5_auth_context *auth_context,
	    const krb5_flags ap_req_options,
	    const char *service,
	    const char *hostname,
	    krb5_data *in_data,
	    krb5_ccache ccache,
	    krb5_data *outbuf)
{
    krb5_error_code ret;
    char **realms;
    char *real_hostname;
    krb5_principal server;

    ret = krb5_expand_hostname_realms (context, hostname,
				       &real_hostname, &realms);
    if (ret)
	return ret;

    ret = krb5_make_principal(context, &server,
			      *realms,
			      service,
			      real_hostname,
			      NULL);
    free (real_hostname);
    krb5_free_host_realm (context, realms);
    if (ret)
	return ret;
    ret = krb5_mk_req_exact (context, auth_context, ap_req_options,
			     server, in_data, ccache, outbuf);
    krb5_free_principal (context, server);
    return ret;
}
