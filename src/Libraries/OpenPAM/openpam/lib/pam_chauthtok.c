/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
 * XSSO 6 page 38
 *
 * Perform password related functions within the PAM framework
 */

int
pam_chauthtok(pam_handle_t *pamh,
	int flags)
{
	int r;

	ENTER();
	if (flags & ~(PAM_SILENT|PAM_CHANGE_EXPIRED_AUTHTOK))
		RETURNC(PAM_SYMBOL_ERR);
	r = openpam_dispatch(pamh, PAM_SM_CHAUTHTOK,
	    flags | PAM_PRELIM_CHECK);
	if (r == PAM_SUCCESS)
		r = openpam_dispatch(pamh, PAM_SM_CHAUTHTOK,
		    flags | PAM_UPDATE_AUTHTOK);
	pam_set_item(pamh, PAM_OLDAUTHTOK, NULL);
	pam_set_item(pamh, PAM_AUTHTOK, NULL);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=openpam_dispatch
 *	=pam_sm_chauthtok
 *	!PAM_IGNORE
 *	PAM_SYMBOL_ERR
 */

/**
 * The =pam_chauthtok function attempts to change the authentication token
 * for the user associated with the pam context specified by the =pamh
 * argument.
 *
 * The =flags argument is the binary or of zero or more of the following
 * values:
 *
 *	=PAM_SILENT:
 *		Do not emit any messages.
 *	=PAM_CHANGE_EXPIRED_AUTHTOK:
 *		Change only those authentication tokens that have expired.
 *
 * If any other bits are set, =pam_chauthtok will return =PAM_SYMBOL_ERR.
 */
