/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include <sys/param.h>

#include <security/pam_appl.h>
#include <security/pam_modules.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.2
 * XSSO 6 page 66
 *
 * Service module implementation for pam_acct_mgmt
 */

int
pam_sm_acct_mgmt(pam_handle_t *pamh,
	int flags,
	int argc,
	const char **argv)
{

	ENTER();
	RETURNC(PAM_SYSTEM_ERR);
}

/*
 * Error codes:
 *
 *	PAM_SERVICE_ERR
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 *	PAM_CONV_ERR
 *	PAM_PERM_DENIED
 *	PAM_IGNORE
 *	PAM_ABORT
 *
 *	PAM_USER_UNKNOWN
 *	PAM_AUTH_ERR
 *	PAM_NEW_AUTHTOK_REQD
 *	PAM_ACCT_EXPIRED
 */

/**
 * The =pam_sm_acct_mgmt function is the service module's implementation
 * of the =pam_acct_mgmt API function.
 */
