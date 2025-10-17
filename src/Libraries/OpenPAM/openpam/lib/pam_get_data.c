/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 43
 *
 * Get module information
 */

int
pam_get_data(const pam_handle_t *pamh,
	const char *module_data_name,
	const void **data)
{
	pam_data_t *dp;

	ENTERS(module_data_name);
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	for (dp = pamh->module_data; dp != NULL; dp = dp->next) {
		if (strcmp(dp->name, module_data_name) == 0) {
			*data = (void *)dp->data;
			RETURNC(PAM_SUCCESS);
		}
	}
	RETURNC(PAM_NO_MODULE_DATA);
}

/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 *	PAM_NO_MODULE_DATA
 */

/**
 * The =pam_get_data function looks up the opaque object associated with
 * the string specified by the =module_data_name argument, in the PAM
 * context specified by the =pamh argument.
 * A pointer to the object is stored in the location pointed to by the
 * =data argument.
 *
 * This function and its counterpart =pam_set_data are useful for managing
 * data that are meaningful only to a particular service module.
 */
