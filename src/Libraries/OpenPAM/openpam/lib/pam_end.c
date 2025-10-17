/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include <stdlib.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 42
 *
 * Terminate the PAM transaction
 */

int
pam_end(pam_handle_t *pamh,
	int status)
{
	pam_data_t *dp;
	int i;

	ENTER();
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* clear module data */
	while ((dp = pamh->module_data) != NULL) {
		if (dp->cleanup)
			(dp->cleanup)(pamh, dp->data, status);
		pamh->module_data = dp->next;
		FREE(dp->name);
		FREE(dp);
	}

	/* clear environment */
	while (pamh->env_count) {
		--pamh->env_count;
		FREE(pamh->env[pamh->env_count]);
	}
	FREE(pamh->env);

	/* clear chains */
	openpam_clear_chains(pamh->chains);

	/* clear items */
	for (i = 0; i < PAM_NUM_ITEMS; ++i)
		pam_set_item(pamh, i, NULL);

	FREE(pamh);

	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 */

/**
 * The =pam_end function terminates a PAM transaction and destroys the
 * corresponding PAM context, releasing all resources allocated to it.
 *
 * The =status argument should be set to the error code returned by the
 * last API call before the call to =pam_end.
 */
