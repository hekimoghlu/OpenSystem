/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#define MIT_KRB5_DEPRECATED 1

#include "heim.h"
#include "mit-krb5.h"
#include <string.h>
#include <errno.h>
#include <syslog.h>

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_setaddrs(mit_krb5_context context,
		       mit_krb5_auth_context ac,
		       mit_krb5_address *caddr,
		       mit_krb5_address *saddr)
{
    LOG_ENTRY();
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getaddrs(mit_krb5_context context,
		       mit_krb5_auth_context ac,
		       mit_krb5_address **caddr,
		       mit_krb5_address **saddr)
{
    LOG_ENTRY();
    *caddr = NULL;
    *saddr = NULL;
    return 0;
}


mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_setports(mit_krb5_context context,
		       mit_krb5_auth_context ac,
		       mit_krb5_address *caddr,
		       mit_krb5_address *saddr)
{
    LOG_ENTRY();
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getkey(mit_krb5_context context,
		     mit_krb5_auth_context ac,
		     mit_krb5_keyblock **keyblock)
{
    LOG_ENTRY();
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_setrcache(mit_krb5_context context,
			mit_krb5_auth_context ac,
			mit_krb5_rcache rcaceh)
{
    LOG_ENTRY();
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getrcache(mit_krb5_context context,
			mit_krb5_auth_context ac,
			mit_krb5_rcache *rcache)
{
    LOG_ENTRY();
    *rcache = NULL;
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getauthenticator(mit_krb5_context context,
			       mit_krb5_auth_context ac,
			       mit_krb5_authenticator **auth)
{
    LOG_ENTRY();
    *auth = NULL;
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getlocalsubkey(mit_krb5_context context,
			     mit_krb5_auth_context ac,
			     mit_krb5_keyblock **key)
{
    LOG_ENTRY();
    krb5_keyblock *hkey = NULL;
    krb5_error_code ret;

    *key = NULL;

    ret = heim_krb5_auth_con_getlocalsubkey(HC(context),
					    (krb5_auth_context)ac,
					    &hkey);
    if (ret)
	return ret;
    if (hkey) {
	*key = mshim_malloc(sizeof(**key));
	mshim_hkeyblock2mkeyblock(hkey, *key);
	heim_krb5_free_keyblock(HC(context), hkey);
    }
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_auth_con_getremotesubkey(mit_krb5_context context,
			      mit_krb5_auth_context ac,
			      mit_krb5_keyblock **key)
{
    LOG_ENTRY();
    krb5_keyblock *hkey = NULL;
    krb5_error_code ret;

    *key = NULL;

    ret = heim_krb5_auth_con_getremotesubkey(HC(context),
					     (krb5_auth_context)ac,
					     &hkey);
    if (ret)
	return ret;

    if (hkey) {
	*key = mshim_malloc(sizeof(**key));
	mshim_hkeyblock2mkeyblock(hkey, *key);
	heim_krb5_free_keyblock(HC(context), hkey);
    }
    return 0;
}
