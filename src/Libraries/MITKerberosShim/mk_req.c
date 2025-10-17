/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#include "mit-krb5.h"
#include <string.h>
#include <errno.h>

static krb5_flags
mshim_mit_ap_req_flags(mit_krb5_flags ap_req_options)
{
    krb5_flags flags = 0;

    if (ap_req_options & MIT_AP_OPTS_USE_SUBKEY)
	flags |= AP_OPTS_USE_SUBKEY;
    if (ap_req_options & MIT_AP_OPTS_USE_SESSION_KEY)
	flags |= AP_OPTS_USE_SESSION_KEY;
    if (ap_req_options & MIT_AP_OPTS_MUTUAL_REQUIRED)
	flags |= AP_OPTS_MUTUAL_REQUIRED;

    return flags;
}



mit_krb5_error_code KRB5_CALLCONV
krb5_mk_req(mit_krb5_context context,
	    mit_krb5_auth_context *ac,
	    mit_krb5_flags ap_req_options,
	    char *service,
	    char *hostname,
	    mit_krb5_data *inbuf,
	    mit_krb5_ccache ccache,
	    mit_krb5_data *outbuf)
{
    krb5_data idata, *d = NULL;
    krb5_data odata;
    krb5_flags flags = mshim_mit_ap_req_flags(ap_req_options);
    krb5_error_code ret;

    LOG_ENTRY();

    memset(outbuf, 0, sizeof(*outbuf));

    if (inbuf) {
	d = &idata;
	idata.data = inbuf->data;
	idata.length = inbuf->length;
    }

    ret = heim_krb5_mk_req(HC(context), (krb5_auth_context *)ac,
			   flags, service, hostname, d,
			   (krb5_ccache)ccache, &odata);
    if (ret == 0)
	mshim_hdata2mdata(&odata, outbuf);

    return ret;
}


mit_krb5_error_code KRB5_CALLCONV
krb5_mk_req_extended(mit_krb5_context context,
		     mit_krb5_auth_context *ac,
		     mit_krb5_flags ap_req_options,
		     mit_krb5_data *inbuf,
		     mit_krb5_creds *cred,
		     mit_krb5_data *outbuf)
 {
    krb5_data idata, *d = NULL;
    krb5_data odata;
    krb5_flags flags = mshim_mit_ap_req_flags(ap_req_options);
    krb5_error_code ret;
    krb5_creds hcreds;

    LOG_ENTRY();

    memset(outbuf, 0, sizeof(*outbuf));

    mshim_mcred2hcred(HC(context), cred, &hcreds);

    if (inbuf) {
	d = &idata;
	idata.data = inbuf->data;
	idata.length = inbuf->length;
    }

    ret = heim_krb5_mk_req_extended(HC(context), (krb5_auth_context *)ac,
				    flags, d, &hcreds, &odata);
    heim_krb5_free_cred_contents(HC(context), &hcreds);
    if (ret == 0)
	mshim_hdata2mdata(&odata, outbuf);

    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_sendauth(mit_krb5_context context,
	      mit_krb5_auth_context *auth_context,
	      mit_krb5_pointer fd,
	      char *appl_version,
	      mit_krb5_principal client,
	      mit_krb5_principal server,
	      mit_krb5_flags ap_req_options,
	      mit_krb5_data *in_data,
	      mit_krb5_creds *in_creds,
	      mit_krb5_ccache ccache,
	      mit_krb5_error **error,
	      mit_krb5_ap_rep_enc_part **rep_result,
	      mit_krb5_creds **out_creds)
{
    mit_krb5_error_code ret;
    struct comb_principal *c = (struct comb_principal *)client;
    struct comb_principal *s = (struct comb_principal *)server;
    krb5_flags flags = mshim_mit_ap_req_flags(ap_req_options);
    krb5_ap_rep_enc_part *hrep_result = NULL;
    krb5_error *herror = NULL;
    krb5_data hin_data;
    krb5_creds hin_creds, *hout_creds = NULL;

    memset(&hin_creds, 0, sizeof(hin_creds));
    heim_krb5_data_zero(&hin_data);

    if (in_data)
	mshim_mdata2hdata(in_data, &hin_data);
    if (in_creds)
	mshim_mcred2hcred(HC(context), in_creds, &hin_creds);

    ret = heim_krb5_sendauth(HC(context),
			     (krb5_auth_context *)auth_context,
			     fd,
			     appl_version,
			     c->heim,
			     s->heim,
			     flags,
			     in_data ? &hin_data : NULL,
			     in_creds ? &hin_creds : NULL,
			     (krb5_ccache)ccache,
			     &herror,
			     &hrep_result,
			     &hout_creds);
    if (in_data)
	free(hin_data.data);
    if (in_creds)
	heim_krb5_free_cred_contents(HC(context), &hin_creds);

    if (herror && error) {
	*error = mshim_malloc(sizeof(**error));
	mshim_herror2merror(HC(context), herror, *error);
    }
    if (hrep_result && rep_result) {
	*rep_result = mshim_malloc(sizeof(**rep_result));
	mshim_haprepencpart2maprepencpart(hrep_result, *rep_result);
    }
    if (hout_creds && out_creds) {
	*out_creds = mshim_malloc(sizeof(**out_creds));
	mshim_hcred2mcred(HC(context), hout_creds, *out_creds);
    }
    if (hrep_result)
	heim_krb5_free_ap_rep_enc_part(HC(context), hrep_result);
    if (herror)
	heim_krb5_free_error(HC(context), herror);
    if (hout_creds)
	heim_krb5_free_creds(HC(context), hout_creds);
    
    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_mk_priv(mit_krb5_context context,
	     mit_krb5_auth_context auth_context,
	     const mit_krb5_data *inbuf,
	     mit_krb5_data *outbuf,
	     mit_krb5_replay_data *replay)
{
    krb5_replay_data outdata;
    mit_krb5_error_code ret;
    krb5_data in, out;

    LOG_ENTRY();

    memset(outbuf, 0, sizeof(*outbuf));
    memset(&outdata, 0, sizeof(outdata));

    in.data = inbuf->data;
    in.length = inbuf->length;

    ret = heim_krb5_mk_priv(HC(context),
			    (krb5_auth_context)auth_context,
			    &in,
			    &out,
			    &outdata);
    if (ret)
	return ret;

    mshim_hdata2mdata(&out, outbuf);
    heim_krb5_data_free(&out);

    if (replay) {
	memset(replay, 0, sizeof(*replay));
	mshim_hreplay2mreplay(&outdata, replay);
    }

    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_mk_safe(mit_krb5_context context,
	     mit_krb5_auth_context auth_context,
	     const mit_krb5_data *inbuf,
	     mit_krb5_data *outbuf,
	     mit_krb5_replay_data *replay)
{
    krb5_replay_data outdata;
    mit_krb5_error_code ret;
    krb5_data in, out;

    LOG_ENTRY();

    memset(outbuf, 0, sizeof(*outbuf));
    memset(&outdata, 0, sizeof(outdata));

    in.data = inbuf->data;
    in.length = inbuf->length;

    ret = heim_krb5_mk_safe(HC(context),
			    (krb5_auth_context)auth_context,
			    &in,
			    &out,
			    &outdata);
    if (ret)
	return ret;

    mshim_hdata2mdata(&out, outbuf);
    heim_krb5_data_free(&out);

    if (replay) {
	memset(replay, 0, sizeof(*replay));
	mshim_hreplay2mreplay(&outdata, replay);
    }

    return 0;
}

