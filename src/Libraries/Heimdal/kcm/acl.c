/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

krb5_error_code
kcm_access(krb5_context context,
	   kcm_client *client,
	   kcm_operation opcode,
	   kcm_ccache ccache)
{
    krb5_error_code ret;

    KCM_ASSERT_VALID(ccache);
    
    HEIMDAL_MUTEX_lock(&ccache->mutex);
    if (ccache->flags & KCM_FLAGS_OWNER_IS_SYSTEM) {
	/* Let root always read system caches */
	if (CLIENT_IS_ROOT(client)) {
	    ret = 0;
	} else {
	    ret = KRB5_FCC_PERM;
	}
    } else if (kcm_is_same_session_locked(client, ccache->uid, ccache->session)) {
	/* same session same as owner */
	ret = 0;
    } else {
	ret = KRB5_FCC_PERM;
    }
    HEIMDAL_MUTEX_unlock(&ccache->mutex);

    if (ret) {
	kcm_log(2, "Process %d is not permitted to call %s on cache %s",
		client->pid, kcm_op2string(opcode), ccache->name);
    }

    return ret;
}

krb5_error_code
kcm_principal_access_locked(krb5_context context,
	   kcm_client *client,
	   krb5_principal server,
	   kcm_operation opcode,
	   kcm_ccache ccache)
{
    KCM_ASSERT_VALID(ccache);
    
    if (!(server->name.name_string.len == 2 &&
	strcmp(server->name.name_string.val[0], "krb5_ccache_conf_data") == 0 &&
	strcmp(server->name.name_string.val[1], "password") == 0))
    {
	// we arent concerned with it, exit and allow access
	return 0;
    }
    
    //default to no access
    krb5_error_code ret = KRB5_FCC_PERM;

    switch (client->iakerb_access) {
	case IAKERB_NOT_CHECKED:
	{
	    const char* callingApp = kcm_client_get_execpath(client);
	    kcm_log(1, "kcm_principal_access: calling app: %s", callingApp);
	    
	    // we check for either the presence of the entitlement or the approved apps.  The file path is the first filter to avoid the expensive code signature check until needed.
	    if (krb5_has_entitlement(client->audit_token, CFSTR("com.apple.private.gssapi.iakerb-data-access"))) {
		kcm_log(1, "kcm_principal_access: has entitlement");
		client->iakerb_access = IAKERB_ACCESS_GRANTED;
	    } else if (strcmp(callingApp, "/System/Library/CoreServices/NetAuthAgent.app/Contents/MacOS/NetAuthSysAgent") == 0 &&
		       krb5_applesigned(context, client->audit_token, "com.apple.NetAuthSysAgent")) {
		client->iakerb_access = IAKERB_ACCESS_GRANTED;
	    } else if (strcmp(callingApp, "/usr/sbin/gssd") == 0 &&
		       krb5_applesigned(context, client->audit_token, "com.apple.gssd")) {
		client->iakerb_access = IAKERB_ACCESS_GRANTED;
	    } else {
		client->iakerb_access = IAKERB_ACCESS_DENIED;
	    }
	    
	    if (client->iakerb_access == IAKERB_ACCESS_GRANTED) {
		ret = 0;
	    }
	    break;
	}
	case IAKERB_ACCESS_GRANTED:
	    ret = 0;
	    break;
	    
	case IAKERB_ACCESS_DENIED:
	    ret = KRB5_FCC_PERM;
	    break;
    }

    kcm_log(1, "kcm_principal_access: access %s", (ret==0 ? "allowed" : "denied"));
    return ret;
}

krb5_error_code
kcm_chmod(krb5_context context,
	  kcm_client *client,
	  kcm_ccache ccache,
	  uint16_t mode)
{
    KCM_ASSERT_VALID(ccache);

    HEIMDAL_MUTEX_lock(&ccache->mutex);
    /* System cache mode can only be set at startup */
    if (ccache->flags & KCM_FLAGS_OWNER_IS_SYSTEM) {
	HEIMDAL_MUTEX_unlock(&ccache->mutex);
	return KRB5_FCC_PERM;
    }

    if (ccache->uid != client->uid) {
	HEIMDAL_MUTEX_unlock(&ccache->mutex);
	return KRB5_FCC_PERM;
    }
    HEIMDAL_MUTEX_unlock(&ccache->mutex);
    return 0;
}

krb5_error_code
kcm_chown(krb5_context context,
	  kcm_client *client,
	  kcm_ccache ccache,
	  uid_t uid)
{
    KCM_ASSERT_VALID(ccache);

    HEIMDAL_MUTEX_lock(&ccache->mutex);
    /* System cache mode can only be set at startup */
    if (ccache->flags & KCM_FLAGS_OWNER_IS_SYSTEM) {
	HEIMDAL_MUTEX_unlock(&ccache->mutex);
	return KRB5_FCC_PERM;
    }

    if (ccache->uid != client->uid) {
	HEIMDAL_MUTEX_unlock(&ccache->mutex);
	return KRB5_FCC_PERM;
    }

    ccache->uid = uid;

    HEIMDAL_MUTEX_unlock(&ccache->mutex);

    return 0;
}

