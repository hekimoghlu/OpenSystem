/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
 * Server-side loopback glue for credentials cache operations; this
 * must be initialized with kcm_internal_ccache(), it is not for real
 * use. This entire file assumes the cache is locked, it does not do
 * any concurrency checking for multithread applications.
 */

#define KCMCACHE(X)	((kcm_ccache)(X)->data.data)
#define CACHENAME(X)	(KCMCACHE(X)->name)

static const char *
kcmss_get_name(krb5_context context,
	       krb5_ccache id)
{
    return CACHENAME(id);
}

static krb5_error_code
kcmss_resolve(krb5_context context, krb5_ccache *id, const char *res)
{
    return KRB5_FCC_INTERNAL;
}

static krb5_error_code
kcmss_gen_new(krb5_context context, krb5_ccache *id)
{
    return KRB5_FCC_INTERNAL;
}

static krb5_error_code
kcmss_initialize(krb5_context context,
		 krb5_ccache id,
		 krb5_principal primary_principal)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = kcm_zero_ccache_data_internal(context, c);
    if (ret) {
    	return ret;
    }

    ret = krb5_copy_principal(context, primary_principal,
			      &c->client);

    return ret;
}

static krb5_error_code
kcmss_close(krb5_context context,
	    krb5_ccache id)
{
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    id->data.data = NULL;
    id->data.length = 0;

    return 0;
}

static krb5_error_code
kcmss_destroy(krb5_context context,
	      krb5_ccache id)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = kcm_ccache_destroy(context, CACHENAME(id));

    return ret;
}

static krb5_error_code
kcmss_store_cred(krb5_context context,
		 krb5_ccache id,
		 krb5_creds *creds)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = kcm_ccache_store_cred_internal(context, c, creds, NULL, 1);

    return ret;
}

static krb5_error_code
kcmss_retrieve(krb5_context context,
	       krb5_ccache id,
	       krb5_flags which,
	       const krb5_creds *mcred,
	       krb5_creds *creds)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);
    krb5_creds *credp;

    KCM_ASSERT_VALID(c);

    ret = kcm_ccache_retrieve_cred_internal(context, c, which,
					    mcred, &credp);
    if (ret)
	return ret;

    ret = krb5_copy_creds_contents(context, credp, creds);
    if (ret)
	return ret;

    return 0;
}

static krb5_error_code
kcmss_get_principal(krb5_context context,
		    krb5_ccache id,
		    krb5_principal *principal)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = krb5_copy_principal(context, c->client,
			      principal);

    return ret;
}

static krb5_error_code
kcmss_get_first (krb5_context context,
		 krb5_ccache id,
		 krb5_cc_cursor *cursor)
{
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    *cursor = c->creds;

    return (*cursor == NULL) ? KRB5_CC_END : 0;
}

static krb5_error_code
kcmss_get_next (krb5_context context,
		krb5_ccache id,
		krb5_cc_cursor *cursor,
		krb5_creds *creds)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = krb5_copy_creds_contents(context,
				   &((struct kcm_creds *)cursor)->cred,
				   creds);
    if (ret)
	return ret;

    *cursor = ((struct kcm_creds *)cursor)->next;
    if (*cursor == 0)
	ret = KRB5_CC_END;

    return ret;
}

static krb5_error_code
kcmss_end_get (krb5_context context,
	       krb5_ccache id,
	       krb5_cc_cursor *cursor)
{
    *cursor = NULL;
    return 0;
}

static krb5_error_code
kcmss_remove_cred(krb5_context context,
		  krb5_ccache id,
		  krb5_flags which,
		  krb5_creds *cred)
{
    krb5_error_code ret;
    kcm_ccache c = KCMCACHE(id);

    KCM_ASSERT_VALID(c);

    ret = kcm_ccache_remove_cred_internal(context, c, which, cred);

    return ret;
}

static krb5_error_code
kcmss_set_flags(krb5_context context,
		krb5_ccache id,
		krb5_flags flags)
{
    return 0;
}

static krb5_error_code
kcmss_get_version(krb5_context context,
		  krb5_ccache id)
{
    return 0;
}

static const krb5_cc_ops krb5_kcmss_ops = {
    KRB5_CC_OPS_VERSION,
    "KCM",
    kcmss_get_name,
    kcmss_resolve,
    kcmss_gen_new,
    kcmss_initialize,
    kcmss_destroy,
    kcmss_close,
    kcmss_store_cred,
    kcmss_retrieve,
    kcmss_get_principal,
    kcmss_get_first,
    kcmss_get_next,
    kcmss_end_get,
    kcmss_remove_cred,
    kcmss_set_flags,
    kcmss_get_version
};

krb5_error_code
kcm_internal_ccache(krb5_context context,
		    kcm_ccache c,
		    krb5_ccache id)
{
    id->ops = &krb5_kcmss_ops;
    id->data.length = sizeof(*c);
    id->data.data = c;

    return 0;
}

