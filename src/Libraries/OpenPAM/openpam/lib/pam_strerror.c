/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include <stdio.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

const char *_pam_err_name[PAM_NUM_ERRORS] = {
	"PAM_SUCCESS",
	"PAM_OPEN_ERR",
	"PAM_SYMBOL_ERR",
	"PAM_SERVICE_ERR",
	"PAM_SYSTEM_ERR",
	"PAM_BUF_ERR",
	"PAM_CONV_ERR",
	"PAM_PERM_DENIED",
	"PAM_MAXTRIES",
	"PAM_AUTH_ERR",
	"PAM_NEW_AUTHTOK_REQD",
	"PAM_CRED_INSUFFICIENT",
	"PAM_AUTHINFO_UNAVAIL",
	"PAM_USER_UNKNOWN",
	"PAM_CRED_UNAVAIL",
	"PAM_CRED_EXPIRED",
	"PAM_CRED_ERR",
	"PAM_ACCT_EXPIRED",
	"PAM_AUTHTOK_EXPIRED",
	"PAM_SESSION_ERR",
	"PAM_AUTHTOK_ERR",
	"PAM_AUTHTOK_RECOVERY_ERR",
	"PAM_AUTHTOK_LOCK_BUSY",
	"PAM_AUTHTOK_DISABLE_AGING",
	"PAM_NO_MODULE_DATA",
	"PAM_IGNORE",
	"PAM_ABORT",
	"PAM_TRY_AGAIN",
	"PAM_MODULE_UNKNOWN",
	"PAM_DOMAIN_UNKNOWN",
	/*
	 * This array only defines strings for OpenPAM error codes.
	 * See below for Apple-specific custom errors, e.g. PAM_APPLE_ACCT_TEMP_LOCK etc..
	 */
};

const char *_pam_apple_err_name[PAM_APPLE_NUM_ERRORS] = {
	"PAM_APPLE_ACCT_TEMP_LOCK",
	"PAM_APPLE_ACCT_LOCKED",
};

/*
 * XSSO 4.2.1
 * XSSO 6 page 92
 *
 * Get PAM standard error message string
 */

const char *
pam_strerror(const pam_handle_t *pamh,
	int error_number)
{
	static char unknown[16];

	(void)pamh;

	switch (error_number) {
	case PAM_SUCCESS:
		return ("success");
	case PAM_OPEN_ERR:
		return ("failed to load module");
	case PAM_SYMBOL_ERR:
		return ("invalid symbol");
	case PAM_SERVICE_ERR:
		return ("error in service module");
	case PAM_SYSTEM_ERR:
		return ("system error");
	case PAM_BUF_ERR:
		return ("memory buffer error");
	case PAM_CONV_ERR:
		return ("conversation failure");
	case PAM_PERM_DENIED:
		return ("permission denied");
	case PAM_MAXTRIES:
		return ("maximum number of tries exceeded");
	case PAM_AUTH_ERR:
		return ("authentication error");
	case PAM_NEW_AUTHTOK_REQD:
		return ("new authentication token required");
	case PAM_CRED_INSUFFICIENT:
		return ("insufficient credentials");
	case PAM_AUTHINFO_UNAVAIL:
		return ("authentication information is unavailable");
	case PAM_USER_UNKNOWN:
		return ("unknown user");
	case PAM_CRED_UNAVAIL:
		return ("failed to retrieve user credentials");
	case PAM_CRED_EXPIRED:
		return ("user credentials have expired");
	case PAM_CRED_ERR:
		return ("failed to set user credentials");
	case PAM_ACCT_EXPIRED:
		return ("user account has expired");
	case PAM_AUTHTOK_EXPIRED:
		return ("password has expired");
	case PAM_SESSION_ERR:
		return ("session failure");
	case PAM_AUTHTOK_ERR:
		return ("authentication token failure");
	case PAM_AUTHTOK_RECOVERY_ERR:
		return ("failed to recover old authentication token");
	case PAM_AUTHTOK_LOCK_BUSY:
		return ("authentication token lock busy");
	case PAM_AUTHTOK_DISABLE_AGING:
		return ("authentication token aging disabled");
	case PAM_NO_MODULE_DATA:
		return ("module data not found");
	case PAM_IGNORE:
		return ("ignore this module");
	case PAM_ABORT:
		return ("general failure");
	case PAM_TRY_AGAIN:
		return ("try again");
	case PAM_MODULE_UNKNOWN:
		return ("unknown module type");
	case PAM_DOMAIN_UNKNOWN:
		return ("unknown authentication domain");
	case PAM_APPLE_ACCT_TEMP_LOCK:
		return ("account temporarily locked");
	case PAM_APPLE_ACCT_LOCKED:
		return ("account locked");
	default:
		snprintf(unknown, sizeof unknown, "#%d", error_number);
		return (unknown);
	}
}

/**
 * The =pam_strerror function returns a pointer to a string containing a
 * textual description of the error indicated by the =error_number
 * argument, in the context of the PAM transaction described by the =pamh
 * argument.
 */
