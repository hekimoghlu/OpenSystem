/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 32
 *
 * Perform PAM account validation procedures
 */

int
pam_acct_mgmt(pam_handle_t *pamh,
	int flags)
{
	int r;

	ENTER();
	r = openpam_dispatch(pamh, PAM_SM_ACCT_MGMT, flags);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=openpam_dispatch
 *	=pam_sm_acct_mgmt
 *	!PAM_IGNORE
 */

/**
 * The =pam_acct_mgmt function verifies and enforces account restrictions
 * after the user has been authenticated.
 *
 * The =flags argument is the binary or of zero or more of the following
 * values:
 *
 *	=PAM_SILENT:
 *		Do not emit any messages.
 *	=PAM_DISALLOW_NULL_AUTHTOK:
 *		Fail if the user's authentication token is null.
 *
 * If any other bits are set, =pam_acct_mgmt will return
 * =PAM_SYMBOL_ERR.
 */
