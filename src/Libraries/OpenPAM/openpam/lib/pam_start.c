/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>		/* getpid() */

#include <System/sys/codesign.h>	/* csops() */
#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 89
 *
 * Initiate a PAM transaction
 */

int
pam_start(const char *service,
	const char *user,
	const struct pam_conv *pam_conv,
	pam_handle_t **pamh)
{
	struct pam_handle *ph;
	int r;

	ENTER();
	if ((ph = calloc(1, sizeof *ph)) == NULL)
		RETURNC(PAM_BUF_ERR);
	if ((r = pam_set_item(ph, PAM_SERVICE, service)) != PAM_SUCCESS)
		goto fail;
	if ((r = pam_set_item(ph, PAM_USER, user)) != PAM_SUCCESS)
		goto fail;
	if ((r = pam_set_item(ph, PAM_CONV, pam_conv)) != PAM_SUCCESS)
		goto fail;

	r = openpam_configure(ph, service);
	if (r == PAM_SYSTEM_ERR && errno == ENOTRECOVERABLE) {
		/* rdar://99495325 (pam_start should not fail because CS_OPS_CLEAR_LV is rejected) */
		int   csflags = 0;
		pid_t pid     = getpid();
		csops(pid, CS_OPS_STATUS, &csflags, sizeof(csflags));
		if ((csflags & CS_INSTALLER) != 0) {
			/* Attempt to load a hard-coded Apple-only (stock) macOS chain. */
			r = openpam_configure_apple(ph, service);
		}
	}
	if (r != PAM_SUCCESS)
		goto fail;

	*pamh = ph;
	openpam_log(PAM_LOG_LIBDEBUG, "pam_start(\"%s\") succeeded", service);
	RETURNC(PAM_SUCCESS);

 fail:
	pam_end(ph, r);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=openpam_configure
 *	=pam_set_item
 *	!PAM_SYMBOL_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =pam_start function creates and initializes a PAM context.
 *
 * The =service argument specifies the name of the policy to apply, and is
 * stored in the =PAM_SERVICE item in the created context.
 *
 * The =user argument specifies the name of the target user - the user the
 * created context will serve to authenticate.
 * It is stored in the =PAM_USER item in the created context.
 *
 * The =pam_conv argument points to a =struct pam_conv describing the
 * conversation function to use; see =pam_conv for details.
 *
 * >pam_get_item
 * >pam_set_item
 * >pam_end
 */
