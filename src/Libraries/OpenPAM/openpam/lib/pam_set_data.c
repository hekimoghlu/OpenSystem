/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 59
 *
 * Set module information
 */

int
pam_set_data(pam_handle_t *pamh,
	const char *module_data_name,
	void *data,
	void (*cleanup)(pam_handle_t *pamh,
		void *data,
		int pam_end_status))
{
	pam_data_t *dp;

	ENTERS(module_data_name);
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	for (dp = pamh->module_data; dp != NULL; dp = dp->next) {
		if (strcmp(dp->name, module_data_name) == 0) {
			if (dp->cleanup)
				(dp->cleanup)(pamh, dp->data, PAM_SUCCESS);
			dp->data = data;
			dp->cleanup = cleanup;
			RETURNC(PAM_SUCCESS);
		}
	}
	if ((dp = malloc(sizeof *dp)) == NULL)
		RETURNC(PAM_BUF_ERR);
	if ((dp->name = strdup(module_data_name)) == NULL) {
		FREE(dp);
		RETURNC(PAM_BUF_ERR);
	}
	dp->data = data;
	dp->cleanup = cleanup;
	dp->next = pamh->module_data;
	pamh->module_data = dp;
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =pam_set_data function associates a pointer to an opaque object
 * with an arbitrary string specified by the =module_data_name argument,
 * in the PAM context specified by the =pamh argument.
 *
 * If not =NULL, the =cleanup argument should point to a function
 * responsible for releasing the resources associated with the object.
 *
 * This function and its counterpart =pam_get_data are useful for managing
 * data that are meaningful only to a particular service module.
 */
