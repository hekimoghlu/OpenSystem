/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
 * XSSO 6 page 34
 *
 * Perform authentication within the PAM framework
 */

int
pam_authenticate(pam_handle_t *pamh,
	int flags)
{
	int r;

	ENTER();
	if (flags & ~(PAM_SILENT|PAM_DISALLOW_NULL_AUTHTOK))
		RETURNC(PAM_SYMBOL_ERR);
	r = openpam_dispatch(pamh, PAM_SM_AUTHENTICATE, flags);
	pam_set_item(pamh, PAM_AUTHTOK, NULL);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=openpam_dispatch
 *	=pam_sm_authenticate
 *	!PAM_IGNORE
 *	PAM_SYMBOL_ERR
 */

/**
 * The =pam_authenticate function attempts to authenticate the user
 * associated with the pam context specified by the =pamh argument.
 *
 * The application is free to call =pam_authenticate as many times as it
 * wishes, but some modules may maintain an internal retry counter and
 * return =PAM_MAXTRIES when it exceeds some preset or hardcoded limit.
 *
 * The =flags argument is the binary or of zero or more of the following
 * values:
 *
 *	=PAM_SILENT:
 *		Do not emit any messages.
 *	=PAM_DISALLOW_NULL_AUTHTOK:
 *		Fail if the user's authentication token is null.
 *
 * If any other bits are set, =pam_authenticate will return
 * =PAM_SYMBOL_ERR.
 */
