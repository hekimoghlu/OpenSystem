/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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

#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * XSSO 4.2.1
 * XSSO 6 page 60
 *
 * Set authentication information
 */

int
pam_set_item(pam_handle_t *pamh,
	int item_type,
	const void *item)
{
	void **slot, *tmp;
	size_t nsize, osize;

	ENTERI(item_type);
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	slot = &pamh->item[item_type];
	osize = nsize = 0;
	switch (item_type) {
	case PAM_SERVICE:
	case PAM_USER:
	case PAM_AUTHTOK:
	case PAM_OLDAUTHTOK:
	case PAM_TTY:
	case PAM_RHOST:
	case PAM_RUSER:
	case PAM_USER_PROMPT:
	case PAM_AUTHTOK_PROMPT:
	case PAM_OLDAUTHTOK_PROMPT:
		if (*slot != NULL)
			osize = strlen(*slot) + 1;
		if (item != NULL)
			nsize = strlen(item) + 1;
		break;
	case PAM_REPOSITORY:
		osize = nsize = sizeof(struct pam_repository);
		break;
	case PAM_CONV:
		osize = nsize = sizeof(struct pam_conv);
		break;
	default:
		RETURNC(PAM_SYMBOL_ERR);
	}
	if (*slot != NULL) {
		memset(*slot, 0xd0, osize);
		FREE(*slot);
	}
	if (item != NULL) {
		if ((tmp = malloc(nsize)) == NULL)
			RETURNC(PAM_BUF_ERR);
		memcpy(tmp, item, nsize);
	} else {
		tmp = NULL;
	}
	*slot = tmp;
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	PAM_SYMBOL_ERR
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =pam_set_item function sets the item specified by the =item_type
 * argument to a copy of the object pointed to by the =item argument.
 * The item is stored in the PAM context specified by the =pamh argument.
 * See =pam_get_item for a list of recognized item types.
 */
