/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "kcm_locl.h"

/*
 * Get a new ticket using a keytab/password and swap it into
 * an existing redentials cache
 */

krb5_error_code
kcm_ccache_acquire_locked(krb5_context context,
		   kcm_ccache ccache,
		   time_t *expire)
{
    krb5_error_code ret = 0;
    krb5_creds cred;
    krb5_const_realm realm;
    krb5_get_init_creds_opt *opt = NULL;
    krb5_ccache_data ccdata;
    char *in_tkt_service = NULL;

    *expire = 0;
    memset(&cred, 0, sizeof(cred));

    KCM_ASSERT_VALID(ccache);

    /* We need a cached key or keytab to acquire credentials */
    if (ccache->flags & KCM_FLAGS_USE_PASSWORD) {
	if (ccache->password == NULL) {
	    krb5_abortx(context,
			"kcm_ccache_acquire: KCM_FLAGS_USE_PASSWORD without password");
	}
    } else if (ccache->flags & KCM_FLAGS_USE_KEYTAB) {
	if (ccache->keytab == NULL) {
	    krb5_abortx(context,
			"kcm_ccache_acquire: KCM_FLAGS_USE_KEYTAB without keytab");
	}
    } else {
	kcm_log(0, "Cannot acquire initial credentials for cache %s without key",
		ccache->name);
	return KRB5_FCC_INTERNAL;
    }

    /* Fake up an internal ccache */
    kcm_internal_ccache(context, ccache, &ccdata);

    /* Now, actually acquire the creds */
    if (ccache->server != NULL) {
	ret = krb5_unparse_name(context, ccache->server, &in_tkt_service);
	if (ret) {
	    kcm_log(0, "Failed to unparse service name for cache %s",
		    ccache->name);
	    return ret;
	}
    }

    realm = krb5_principal_get_realm(context, ccache->client);

    ret = krb5_get_init_creds_opt_alloc(context, &opt);
    if (ret)
	goto out;
    krb5_get_init_creds_opt_set_default_flags(context, "kcm", realm, opt);
    if (ccache->tkt_life != 0)
	krb5_get_init_creds_opt_set_tkt_life(opt, ccache->tkt_life);
    if (ccache->renew_life != 0)
	krb5_get_init_creds_opt_set_renew_life(opt, ccache->renew_life);

    if (ccache->flags & KCM_FLAGS_USE_PASSWORD) {
	ret = krb5_get_init_creds_password(context,
					   &cred,
					   ccache->client,
					   ccache->password,
					   NULL,
					   NULL,
					   0,
					   in_tkt_service,
					   opt);
    } else {
	ret = krb5_get_init_creds_keytab(context,
					 &cred,
					 ccache->client,
					 ccache->keytab,
					 0,
					 in_tkt_service,
					 opt);
    }

    if (ret) {
	const char *msg = krb5_get_error_message(context, ret);
	kcm_log(0, "Failed to acquire credentials for cache %s: %s",
		ccache->name, msg);
	krb5_free_error_message(context, msg);
	if (in_tkt_service != NULL)
	    free(in_tkt_service);
	goto out;
    }

    if (in_tkt_service != NULL)
	free(in_tkt_service);

    /* Swap them in */
    kcm_ccache_remove_creds_internal(context, ccache);

    ret = kcm_ccache_store_cred_internal(context, ccache, &cred, NULL, 0);
    if (ret) {
	const char *msg = krb5_get_error_message(context, ret);
	kcm_log(0, "Failed to store credentials for cache %s: %s",
		ccache->name, msg);
	krb5_free_error_message(context, msg);
	krb5_free_cred_contents(context, &cred);
	goto out;
    }

    *expire = cred.times.endtime;

out:
    if (opt)
	krb5_get_init_creds_opt_free(context, opt);

    return ret;
}
