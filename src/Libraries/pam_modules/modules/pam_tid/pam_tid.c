/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
/******************************************************************
 * The purpose of this module is to provide a Touch ID
 * based authentication module for Mac OS X.
 ******************************************************************/

#include <CoreFoundation/CoreFoundation.h>
#include <coreauthd_spi.h>
#include <pwd.h>
#include <LocalAuthentication/LAPrivateDefines.h>

#define PAM_SM_AUTH
#define PAM_SM_ACCOUNT

#include <security/pam_modules.h>
#include <security/pam_appl.h>
#include <Security/Authorization.h>
#include <vproc_priv.h>
#include "Logging.h"
#include "Common.h"

#ifdef PAM_USE_OS_LOG
PAM_DEFINE_LOG(touchid)
#define PAM_LOG PAM_LOG_touchid()
#endif

PAM_EXTERN int
pam_sm_authenticate(pam_handle_t * pamh, int flags, int argc, const char **argv)
{
    _LOG_DEBUG("pam_tid: pam_sm_authenticate");

    int retval = PAM_AUTH_ERR;
    CFTypeRef context = NULL;
    CFErrorRef error = NULL;
    CFMutableDictionaryRef options = NULL;
    CFNumberRef key = NULL;
    CFNumberRef value = NULL;
	CFNumberRef key2 = NULL;
	CFNumberRef value2 = NULL;
	AuthorizationRef authorizationRef = NULL;

    int tmp;

    const char *user = NULL;
    struct passwd *pwd = NULL;
    struct passwd pwdbuf;

    /* determine the required bufsize for getpwnam_r */
    long bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (bufsize == -1) {
        bufsize = 2 * PATH_MAX;
    }
    
    /* get information about user to authenticate for */
    char *buffer = malloc(bufsize);
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || !user ||
        getpwnam_r(user, &pwdbuf, buffer, bufsize, &pwd) != 0 || !pwd) {
        _LOG_ERROR("unable to obtain the username.");
        retval = PAM_AUTHINFO_UNAVAIL;
        goto cleanup;
    }

	// check if we are running under Aqua session
	char *manager;
	if (vproc_swap_string(NULL, VPROC_GSK_MGR_NAME, NULL, &manager) != NULL) {
		_LOG_ERROR("unable to determine session.");
		retval = PAM_AUTH_ERR;
		goto cleanup;
	}
	bool runningInAquaSession = manager ? !strcmp(manager, VPROCMGR_SESSION_AQUA) : FALSE;
	free(manager);
	if (!runningInAquaSession) {
		_LOG_ERROR("UI not available.");
		retval = PAM_AUTH_ERR;
		goto cleanup;
	}

	// check if user is eligible to use Touch ID. If not, fail.
    /* prepare the options dictionary, aka rewrite @{ @(LAOptionNotInteractive) : @YES } without Foundation */
    tmp = kLAOptionNotInteractive;
    key = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &tmp);

    tmp = 1;
    value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &tmp);

	tmp = kLAOptionUserId;
	key2 = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &tmp);

	tmp = pwd->pw_uid;
	value2 = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &tmp);

	if (! (key && value && key2 && value2)) {
		_LOG_ERROR("unable to create data structures.");
		retval = PAM_AUTH_ERR;
		goto cleanup;
	}

    options = CFDictionaryCreateMutable(kCFAllocatorDefault, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(options, key, value);
	CFDictionarySetValue(options, key2, value2);

	context = LACreateNewContextWithACMContext(NULL, &error);
	if (!context) {
		_LOG_ERROR("unable to create context.");
		retval = PAM_AUTH_ERR;
		goto cleanup;
	}

    /* evaluate policy */
    if (!LAEvaluatePolicy(context, kLAPolicyDeviceOwnerAuthenticationWithBiometrics, options, &error)) {
		// error is intended as failure means Touch ID is not usable which is in fact not an error but the state we need to handle
		if (CFErrorGetCode(error) != kLAErrorNotInteractive) {
			_LOG_DEBUG("policy evaluation failed: %ld", CFErrorGetCode(error));
			retval = PAM_AUTH_ERR;
			goto cleanup;
		}
    }

	OSStatus status = AuthorizationCreate(NULL, NULL, kAuthorizationFlagDefaults, &authorizationRef);
	if (status == errAuthorizationSuccess) {
		AuthorizationItem myItems = {"com.apple.security.sudo", 0, NULL, 0};
		AuthorizationRights myRights = {1, &myItems};
		AuthorizationRights *authorizedRights = NULL;
		AuthorizationFlags flags = kAuthorizationFlagDefaults | kAuthorizationFlagInteractionAllowed | kAuthorizationFlagExtendRights;
		status = AuthorizationCopyRights(authorizationRef, &myRights, kAuthorizationEmptyEnvironment, flags, &authorizedRights);
		_LOG_DEBUG("Authorization result: %d", (int)status);
		if (authorizedRights)
			AuthorizationFreeItemSet(authorizedRights);
		AuthorizationFree(authorizationRef, kAuthorizationFlagDefaults);
	}

    /* we passed the Touch ID authentication successfully */
	if (status == errAuthorizationSuccess) {
		retval = PAM_SUCCESS;
	}

cleanup:
	CFReleaseSafe(context);
	CFReleaseSafe(key);
	CFReleaseSafe(value);
	CFReleaseSafe(key2);
	CFReleaseSafe(value2);
	CFReleaseSafe(options);
	CFReleaseSafe(error);
    free(buffer);
	_LOG_DEBUG("pam_tid: pam_sm_authenticate returned %d", retval);
    return retval;
}


PAM_EXTERN int
pam_sm_setcred(pam_handle_t * pamh, int flags, int argc, const char **argv)
{
    return PAM_SUCCESS;
}


PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t * pamh, int flags, int argc, const char **argv)
{
    return PAM_SUCCESS;
}
