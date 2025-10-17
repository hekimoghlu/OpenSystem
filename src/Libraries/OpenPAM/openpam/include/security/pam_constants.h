/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#ifndef SECURITY_PAM_CONSTANTS_H_INCLUDED
#define SECURITY_PAM_CONSTANTS_H_INCLUDED

#include <security/openpam_version.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * XSSO 5.2
 */
enum {
	PAM_SUCCESS			=   0,
	PAM_OPEN_ERR			=   1,
	PAM_SYMBOL_ERR			=   2,
	PAM_SERVICE_ERR			=   3,
	PAM_SYSTEM_ERR			=   4,
	PAM_BUF_ERR			=   5,
	PAM_CONV_ERR			=   6,
	PAM_PERM_DENIED			=   7,
	PAM_MAXTRIES			=   8,
	PAM_AUTH_ERR			=   9,
	PAM_NEW_AUTHTOK_REQD		=  10,
	PAM_CRED_INSUFFICIENT		=  11,
	PAM_AUTHINFO_UNAVAIL		=  12,
	PAM_USER_UNKNOWN		=  13,
	PAM_CRED_UNAVAIL		=  14,
	PAM_CRED_EXPIRED		=  15,
	PAM_CRED_ERR			=  16,
	PAM_ACCT_EXPIRED		=  17,
	PAM_AUTHTOK_EXPIRED		=  18,
	PAM_SESSION_ERR			=  19,
	PAM_AUTHTOK_ERR			=  20,
	PAM_AUTHTOK_RECOVERY_ERR	=  21,
	PAM_AUTHTOK_LOCK_BUSY		=  22,
	PAM_AUTHTOK_DISABLE_AGING	=  23,
	PAM_NO_MODULE_DATA		=  24,
	PAM_IGNORE			=  25,
	PAM_ABORT			=  26,
	PAM_TRY_AGAIN			=  27,
	PAM_MODULE_UNKNOWN		=  28,
	PAM_DOMAIN_UNKNOWN		=  29,
	PAM_NUM_ERRORS,					/* OpenPAM extension */

	/* CUSTOM APPLE OPENPAM ERROR CODES: START */
	PAM_APPLE_MIN_ERROR      = 1024,
	PAM_APPLE_ACCT_TEMP_LOCK = 1024,
	PAM_APPLE_ACCT_LOCKED    = 1025,
    PAM_APPLE_KEK_ERROR      = 1026,
    PAM_APPLE_WRONG_CARD     = 1027,
	/* Insert new custom Apple error codes above! */
	PAM_APPLE_MAX_ERROR,
	PAM_APPLE_NUM_ERRORS     = PAM_APPLE_MAX_ERROR - PAM_APPLE_MIN_ERROR
	/* CUSTOM APPLE OPENPAM ERROR CODES: END */
};

/*
 * XSSO 5.3
 */
enum {
	PAM_PROMPT_ECHO_OFF		=   1,
	PAM_PROMPT_ECHO_ON		=   2,
	PAM_ERROR_MSG			=   3,
	PAM_TEXT_INFO			=   4,
	PAM_MAX_NUM_MSG			=  32,
	PAM_MAX_MSG_SIZE		= 512,
	PAM_MAX_RESP_SIZE		= 512
};

/*
 * XSSO 5.4
 */
enum {
	/* some compilers promote 0x8000000 to long */
	PAM_SILENT			= (-0x7fffffff - 1),
	PAM_DISALLOW_NULL_AUTHTOK	= 0x1,
	PAM_ESTABLISH_CRED		= 0x1,
	PAM_DELETE_CRED			= 0x2,
	PAM_REINITIALIZE_CRED		= 0x4,
	PAM_REFRESH_CRED		= 0x8,
	PAM_PRELIM_CHECK		= 0x1,
	PAM_UPDATE_AUTHTOK		= 0x2,
	PAM_CHANGE_EXPIRED_AUTHTOK	= 0x4
};

/*
 * XSSO 5.5
 */
enum {
	PAM_SERVICE			=   1,
	PAM_USER			=   2,
	PAM_TTY				=   3,
	PAM_RHOST			=   4,
	PAM_CONV			=   5,
	PAM_AUTHTOK			=   6,
	PAM_OLDAUTHTOK			=   7,
	PAM_RUSER			=   8,
	PAM_USER_PROMPT			=   9,
	PAM_REPOSITORY			=  10,
	PAM_AUTHTOK_PROMPT		=  11,		/* OpenPAM extension */
	PAM_OLDAUTHTOK_PROMPT		=  12,		/* OpenPAM extension */
	PAM_NUM_ITEMS					/* OpenPAM extension */
};

#ifdef __cplusplus
}
#endif

#endif /* !SECURITY_PAM_CONSTANTS_H_INCLUDED */
