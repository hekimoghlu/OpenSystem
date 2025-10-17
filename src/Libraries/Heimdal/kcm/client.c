/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include <pwd.h>

krb5_error_code
kcm_ccache_resolve_client(krb5_context context,
			  kcm_client *client,
			  kcm_operation opcode,
			  const char *name,
			  kcm_ccache *ccache)
{
    krb5_error_code ret;

    ret = kcm_ccache_resolve_by_name(context, name, ccache);
    if (ret) {
	kcm_log(1, "Failed to resolve cache %s", name);
	return ret;
    }

    ret = kcm_access(context, client, opcode, *ccache);
    if (ret) {
	ret = KRB5_FCC_NOFILE; /* don't disclose */
	kcm_release_ccache(context, *ccache);
    }

    return ret;
}

krb5_error_code
kcm_ccache_destroy_client(krb5_context context,
			  kcm_client *client,
			  const char *name)
{
    krb5_error_code ret;
    kcm_ccache ccache;

    ret = kcm_ccache_resolve_by_name(context, name, &ccache);
    if (ret) {
	kcm_log(1, "Failed to resolve cache %s", name);
	return ret;
    }

    ret = kcm_access(context, client, KCM_OP_DESTROY, ccache);
    kcm_release_ccache(context, ccache);
    if (ret)
	return ret;

    return kcm_ccache_destroy(context, name);
}

krb5_error_code
kcm_ccache_new_client(krb5_context context,
		      kcm_client *client,
		      const char *name,
		      kcm_ccache *ccache_p)
{
    krb5_error_code ret;
    kcm_ccache ccache;

    ret = kcm_ccache_resolve_by_name(context, name, &ccache);
    if (ret == 0) {
	if ((ccache->uid != client->uid) && !CLIENT_IS_ROOT(client))
	    return KRB5_FCC_PERM;
    } else if (ret != KRB5_FCC_NOFILE && !(CLIENT_IS_ROOT(client) && ret == KRB5_FCC_PERM)) {
		return ret;
    }

    if (ret == KRB5_FCC_NOFILE) {
	ret = kcm_ccache_new_internal(context, name, client->uid, client->session, &ccache);
	if (ret) {
	    kcm_log(1, "Failed to initialize cache %s", name);
	    return ret;
	}

	/* 
	 * add notification when the session goes away, so we can
	 * remove the credential
	 */
	kcm_session_add(client->session);

    } else {
	ret = kcm_zero_ccache_data(context, ccache);
	if (ret) {
	    kcm_log(1, "Failed to empty cache %s", name);
	    kcm_release_ccache(context, ccache);
	    return ret;
	}
	heim_ipc_event_cancel(ccache->renew_event);
	heim_ipc_event_cancel(ccache->expire_event);
    }

    ret = kcm_access(context, client, KCM_OP_INITIALIZE, ccache);
    if (ret) {
	kcm_release_ccache(context, ccache);
	kcm_ccache_destroy(context, name);
	return ret;
    }

    /*
     * Finally, if the user is root and the cache was created under
     * another user's name, chown the cache to that user.
     */
    if (CLIENT_IS_ROOT(client)) {
	unsigned long uid;
	int matches = sscanf(name,"%ld:",&uid);
	if (matches == 0)
	    matches = sscanf(name,"%ld",&uid);
	if (matches == 1) {
	    kcm_chown(context, client, ccache, (uid_t)uid);
	}
    }

    *ccache_p = ccache;
    return 0;
}

const char *
kcm_client_get_execpath(kcm_client *client)
{
    if (client->execpath[0] == '\0') {
	int ret = proc_pidpath(client->pid, client->execpath, sizeof(client->execpath));
	if (ret != -1)
	    client->execpath[sizeof(client->execpath) - 1] = '\0';
	else {
	    /* failed, lets not try again */
	    client->execpath[0] = 0x01;
	    client->execpath[1] = 0x0;
	}
    }
    if (client->execpath[0] != '/')
	return NULL;

    return client->execpath;
}

krb5_boolean
krb5_has_entitlement(audit_token_t token, CFStringRef entitlement)
{
    
    SecTaskRef task = SecTaskCreateWithAuditToken(NULL, token);
    if (!task) {
	kcm_log(1, "unable to create task for audit token");
	return false;
    }
    
    CFErrorRef error = NULL;
    CFTypeRef hasEntitlement = SecTaskCopyValueForEntitlement(task, entitlement, &error);
    CFRELEASE_NULL(task);
    if (hasEntitlement == NULL || CFGetTypeID(hasEntitlement) != CFBooleanGetTypeID() || !CFBooleanGetValue(hasEntitlement)) {
	if (error) {
	    CFStringRef errorDescription = CFErrorCopyFailureReason(error);
	    kcm_log(1, "error retrieving entitlement: %ld, %s", (long)CFErrorGetCode(error), CFStringGetCStringPtr(errorDescription, kCFStringEncodingUTF8));
	    CFRELEASE_NULL(errorDescription);
	    CFRELEASE_NULL(error);
	}
	CFRELEASE_NULL(hasEntitlement);
	return false;
    }
    
    CFRELEASE_NULL(hasEntitlement);
    return true;
    
}

krb5_boolean
krb5_applesigned(krb5_context context, audit_token_t auditToken, const char *identifierToVerify)
{
    bool applesigned = false;
    OSStatus result = noErr;
    CFDictionaryRef attributes = NULL;
    SecCodeRef codeRef = NULL;
    SecRequirementRef secRequirementRef = NULL;
    CFStringRef requirement = NULL;
    
    requirement = CFStringCreateWithFormat(NULL, NULL, CFSTR("identifier \"%s\" and anchor apple"), identifierToVerify);
    kcm_log(1, "requirement: %s", CFStringGetCStringPtr(requirement, kCFStringEncodingUTF8));
    result = SecRequirementCreateWithString(requirement, kSecCSDefaultFlags, &secRequirementRef);
    if (result || !secRequirementRef) {
	kcm_log(1, "Error creating requirement %d ", result);
	applesigned = false;
	goto cleanup;
    }
        
    const void *keys[] = {
	kSecGuestAttributeAudit,
    };
    const void *values[] = {
	CFDataCreate(NULL, (const UInt8*)&auditToken, sizeof(auditToken)),
    };
    
    attributes = CFDictionaryCreate(NULL, keys, values, 1,
						    &kCFTypeDictionaryKeyCallBacks, & kCFTypeDictionaryValueCallBacks);
    
    result = SecCodeCopyGuestWithAttributes(NULL, attributes, kSecCSDefaultFlags, &codeRef);
    if (result || !codeRef) {
	kcm_log(1, "Error creating code ref: %d ", result);
	applesigned = false;
	goto cleanup;
    }
    
    result = SecCodeCheckValidity(codeRef, kSecCSDefaultFlags, secRequirementRef);
    if (result) {
	kcm_log(1, "Error checking requirement: %d ", result);
	applesigned = false;
	goto cleanup;
    }
    
    applesigned = true;
    
cleanup:
    
    CFRELEASE_NULL(requirement);
    CFRELEASE_NULL(secRequirementRef);
    CFRELEASE_NULL(attributes);
    CFRELEASE_NULL(codeRef);
    
    return applesigned;
}
