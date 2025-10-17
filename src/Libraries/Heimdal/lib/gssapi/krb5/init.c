/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#include "gsskrb5_locl.h"

heim_string_t _gsskrb5_kGSSICPassword;
heim_string_t _gsskrb5_kGSSICKerberosCacheName;
heim_string_t _gsskrb5_kGSSICSiteName;
heim_string_t _gsskrb5_kGSSICCertificate;
heim_string_t _gsskrb5_kGSSICLKDCHostname;
heim_string_t _gsskrb5_kGSSICAppIdentifierACL;
heim_string_t _gsskrb5_kGSSICAppleSourceApp;
heim_string_t _gsskrb5_kGSSICAppleSourceAppAuditToken;
heim_string_t _gsskrb5_kGSSICAppleSourceAppPID;
heim_string_t _gsskrb5_kGSSICAppleSourceAppUUID;
heim_string_t _gsskrb5_kGSSICAppleSourceAppSigningIdentity;
heim_string_t _gsskrb5_kGSSICVerifyCredential;
heim_string_t _gsskrb5_kGSSICVerifyCredentialAcceptorName;
heim_string_t _gsskrb5_kGSSICCreateNewCredential;
heim_string_t _gsskrb5_kGSSICAuthenticationContext;

static HEIMDAL_thread_key context_key;

static void
destroy_context(void *ptr)
{
    krb5_context context = ptr;

    if (context == NULL)
	return;
    krb5_free_context(context);
}

static void
once_func(void *ctx)
{
    int ret;

    _gsskrb5_kGSSICPassword = heim_string_create("kGSSICPassword");
    _gsskrb5_kGSSICCertificate = heim_string_create("kGSSICCertificate");
    _gsskrb5_kGSSICSiteName = heim_string_create("kGSSICSiteName");
    _gsskrb5_kGSSICKerberosCacheName = heim_string_create("kGSSICKerberosCacheName");
    _gsskrb5_kGSSICLKDCHostname = heim_string_create("kGSSICLKDCHostname");
    _gsskrb5_kGSSICAppIdentifierACL = heim_string_create("kGSSICAppIdentifierACL");
    _gsskrb5_kGSSICAppleSourceApp = heim_string_create("kGSSICAppleSourceApp");
    _gsskrb5_kGSSICAppleSourceAppAuditToken = heim_string_create("kGSSICAppleSourceAppAuditToken");
    _gsskrb5_kGSSICAppleSourceAppPID = heim_string_create("kGSSICAppleSourceAppPID");
    _gsskrb5_kGSSICAppleSourceAppUUID = heim_string_create("kGSSICAppleSourceAppUUID");
    _gsskrb5_kGSSICAppleSourceAppSigningIdentity = heim_string_create("kGSSICAppleSourceAppSigningIdentity");
    _gsskrb5_kGSSICVerifyCredential = heim_string_create("kGSSICVerifyCredential");
    _gsskrb5_kGSSICVerifyCredentialAcceptorName = heim_string_create("kGSSICVerifyCredentialAcceptorName");
    _gsskrb5_kGSSICCreateNewCredential = heim_string_create("kGSSICCreateNewCredential");
    _gsskrb5_kGSSICAuthenticationContext = heim_string_create("kGSSICAuthenticationContext");
    
    HEIMDAL_key_create(&context_key, destroy_context, ret);
}

krb5_error_code
_gsskrb5_init (krb5_context *context)
{
    static heim_base_once_t once;
    krb5_error_code ret = 0;

    heim_base_once_f(&once, NULL, once_func);

    *context = HEIMDAL_getspecific(context_key);
    if (*context == NULL) {

	ret = krb5_init_context(context);
	if (ret == 0) {
	    HEIMDAL_setspecific(context_key, *context, ret);
	    if (ret) {
		krb5_free_context(*context);
		*context = NULL;
	    }
	}
    } else {
	krb5_reload_config(*context, 0, NULL);
    }

    return ret;
}
