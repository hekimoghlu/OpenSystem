/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_default(mit_krb5_context context, mit_krb5_ccache *cache)
{
    LOG_ENTRY();
    return heim_krb5_cc_default(HC(context), (krb5_ccache *)cache);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_resolve(mit_krb5_context context, const char *str, mit_krb5_ccache *cache)
{
    LOG_ENTRY();
    return heim_krb5_cc_resolve(HC(context), str, (krb5_ccache *)cache);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_initialize(mit_krb5_context context,
		   mit_krb5_ccache cache,
		   mit_krb5_principal principal)
{
    struct comb_principal *p = (struct comb_principal *)principal;
    LOG_ENTRY();
    return heim_krb5_cc_initialize(HC(context), (krb5_ccache)cache, p->heim);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_store_cred(mit_krb5_context context,
		   mit_krb5_ccache cache,
		   mit_krb5_creds *creds)
{
    krb5_error_code ret;
    krb5_creds hcred;
    LOG_ENTRY();
    mshim_mcred2hcred(HC(context), creds, &hcred);
    ret = heim_krb5_cc_store_cred(HC(context), (krb5_ccache)cache, &hcred);
    heim_krb5_free_cred_contents(HC(context), &hcred);
    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_get_principal (mit_krb5_context context,
		       mit_krb5_ccache cache,
		       mit_krb5_principal *principal)
{
    krb5_principal p;
    krb5_error_code ret;

    LOG_ENTRY();

    ret = heim_krb5_cc_get_principal(HC(context), (krb5_ccache)cache, &p);
    if (ret)
	return ret;
    *principal = mshim_hprinc2mprinc(HC(context), p);
    heim_krb5_free_principal(HC(context), p);
    if (*principal == NULL) {
	krb5_set_error_message(context, ENOMEM, "out of memory");
	return ENOMEM;
    }
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_close(mit_krb5_context context,
	      mit_krb5_ccache cache)
{
    return heim_krb5_cc_close(HC(context), (krb5_ccache)cache);
}

const char * KRB5_CALLCONV
krb5_cc_get_name (mit_krb5_context context, mit_krb5_ccache cache)
{
    return heim_krb5_cc_get_name(HC(context), (krb5_ccache)cache);
}

const char * KRB5_CALLCONV
krb5_cc_get_type (mit_krb5_context context, mit_krb5_ccache cache)
{
    return heim_krb5_cc_get_type(HC(context), (krb5_ccache)cache);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_get_config(mit_krb5_context context, mit_krb5_ccache id,
		   mit_krb5_const_principal principal,
		   const char *key, mit_krb5_data *data)
{
    struct comb_principal *p = (struct comb_principal *)principal;
    krb5_principal hc = NULL;
    krb5_error_code ret;
    krb5_data hdata;

    if (p)
	hc = p->heim;

    ret = heim_krb5_cc_get_config(HC(context), (krb5_ccache)id, hc, key, &hdata);
    if (ret)
	return ret;
    ret = mshim_hdata2mdata(&hdata, data);
    heim_krb5_data_free(&hdata);
    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_new_unique(mit_krb5_context context,
		   const char *type,
		   const char *hint,
		   mit_krb5_ccache *id)
{
    LOG_ENTRY();
    return heim_krb5_cc_new_unique(HC(context), type, hint, (krb5_ccache *)id);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_gen_new (mit_krb5_context context, mit_krb5_ccache *id)
{
    LOG_ENTRY();
    return heim_krb5_cc_new_unique(HC(context), NULL, NULL, (krb5_ccache *)id);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_cache_match (mit_krb5_context context,
		     mit_krb5_principal client,
		     mit_krb5_ccache *id)
{
    struct comb_principal *p = (struct comb_principal *)client;
    return heim_krb5_cc_cache_match(HC(context), p->heim, (krb5_ccache *)id);
}

static const struct mshim_map_flags whichfields_flags[] = {
    { MIT_KRB5_TC_MATCH_TIMES,		KRB5_TC_MATCH_TIMES },
    { MIT_KRB5_TC_MATCH_IS_SKEY,	KRB5_TC_MATCH_IS_SKEY },
    { MIT_KRB5_TC_MATCH_FLAGS,		KRB5_TC_MATCH_FLAGS },
    { MIT_KRB5_TC_MATCH_TIMES_EXACT,	KRB5_TC_MATCH_TIMES_EXACT },
    { MIT_KRB5_TC_MATCH_FLAGS_EXACT,	KRB5_TC_MATCH_FLAGS_EXACT },
    { MIT_KRB5_TC_MATCH_AUTHDATA,	KRB5_TC_MATCH_AUTHDATA },
    { MIT_KRB5_TC_MATCH_SRV_NAMEONLY,	KRB5_TC_MATCH_SRV_NAMEONLY },
    { MIT_KRB5_TC_MATCH_2ND_TKT,	KRB5_TC_MATCH_2ND_TKT },
    { MIT_KRB5_TC_MATCH_KTYPE,		KRB5_TC_MATCH_KEYTYPE },
    { MIT_KRB5_TC_SUPPORTED_KTYPES,	0 },
    { 0 }
};

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_retrieve_cred(mit_krb5_context context,
		      mit_krb5_ccache cache,
		      mit_krb5_flags flags,
		      mit_krb5_creds *mcreds,
		      mit_krb5_creds *creds)
{
    krb5_error_code ret;
    krb5_creds hcreds, hmcreds;
    krb5_flags whichfields;

    LOG_ENTRY();

    memset(creds, 0, sizeof(*creds));
    memset(&hcreds, 0, sizeof(hcreds));

    whichfields = (unsigned int)mshim_remap_flags(flags, whichfields_flags);

    mshim_mcred2hcred(HC(context), mcreds, &hmcreds);
    ret = heim_krb5_cc_retrieve_cred(HC(context), (krb5_ccache)cache, whichfields,
				     &hmcreds, &hcreds);
    heim_krb5_free_cred_contents(HC(context), &hmcreds);
    if (ret == 0) {
	mshim_hcred2mcred(HC(context), &hcreds, creds);
	heim_krb5_free_cred_contents(HC(context), &hcreds);
    }
    return ret;

}

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_next_cred(mit_krb5_context context,
		  mit_krb5_ccache cache,
		  mit_krb5_cc_cursor *cursor,
		  mit_krb5_creds *creds)
{
    krb5_error_code ret;
    krb5_creds c;

    LOG_ENTRY();

    ret = heim_krb5_cc_next_cred(HC(context), (krb5_ccache)cache, (krb5_cc_cursor *)cursor, &c);
    if (ret == 0) {
	mshim_hcred2mcred(HC(context), &c, creds);
	heim_krb5_free_cred_contents(HC(context), &c);
    }
    return ret;
}

/* <rdar://problem/7381784> */

mit_krb5_error_code KRB5_CALLCONV
krb5_cc_end_seq_get (mit_krb5_context context, mit_krb5_ccache cache,
		     mit_krb5_cc_cursor *cursor)
{
    LOG_ENTRY();
    if (context == NULL || cache == NULL || cursor == NULL || *cursor == NULL) {
	/* 
	 * We have to return a non failure code to make AppleConnect
	 * happy, when testing this, make sure you don't have any
	 * credentails at all since AC will pick up both API and FILE
	 * based credentials.
	 */
	return 0;
    }
    return heim_krb5_cc_end_seq_get(HC(context), (krb5_ccache)cache, (krb5_cc_cursor)cursor);
}
