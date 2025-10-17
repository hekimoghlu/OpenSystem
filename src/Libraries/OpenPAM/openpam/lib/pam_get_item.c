/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

const char *_pam_item_name[PAM_NUM_ITEMS] = {
	"(NO ITEM)",
	"PAM_SERVICE",
	"PAM_USER",
	"PAM_TTY",
	"PAM_RHOST",
	"PAM_CONV",
	"PAM_AUTHTOK",
	"PAM_OLDAUTHTOK",
	"PAM_RUSER",
	"PAM_USER_PROMPT",
	"PAM_REPOSITORY",
	"PAM_AUTHTOK_PROMPT",
	"PAM_OLDAUTHTOK_PROMPT"
};

/*
 * XSSO 4.2.1
 * XSSO 6 page 46
 *
 * Get PAM information
 */

int
pam_get_item(const pam_handle_t *pamh,
	int item_type,
	const void **item)
{

	ENTERI(item_type);
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	switch (item_type) {
	case PAM_SERVICE:
	case PAM_USER:
	case PAM_AUTHTOK:
	case PAM_OLDAUTHTOK:
	case PAM_TTY:
	case PAM_RHOST:
	case PAM_RUSER:
	case PAM_CONV:
	case PAM_USER_PROMPT:
	case PAM_AUTHTOK_PROMPT:
	case PAM_OLDAUTHTOK_PROMPT:
	case PAM_REPOSITORY:
		*item = pamh->item[item_type];
		RETURNC(PAM_SUCCESS);
	default:
		RETURNC(PAM_SYMBOL_ERR);
	}
}

/*
 * Error codes:
 *
 *	PAM_SYMBOL_ERR
 *	PAM_SYSTEM_ERR
 */

/**
 * The =pam_get_item function stores a pointer to the item specified by
 * the =item_type argument in the location specified by the =item
 * argument.
 * The item is retrieved from the PAM context specified by the =pamh
 * argument.
 * The following item types are recognized:
 *
 *	=PAM_SERVICE:
 *		The name of the requesting service.
 *	=PAM_USER:
 *		The name of the user the application is trying to
 *		authenticate.
 *	=PAM_TTY:
 *		The name of the current terminal.
 *	=PAM_RHOST:
 *		The name of the applicant's host.
 *	=PAM_CONV:
 *		A =struct pam_conv describing the current conversation
 *		function.
 *	=PAM_AUTHTOK:
 *		The current authentication token.
 *	=PAM_OLDAUTHTOK:
 *		The expired authentication token.
 *	=PAM_RUSER:
 *		The name of the applicant.
 *	=PAM_USER_PROMPT:
 *		The prompt to use when asking the applicant for a user
 *		name to authenticate as.
 *	=PAM_AUTHTOK_PROMPT:
 *		The prompt to use when asking the applicant for an
 *		authentication token.
 *	=PAM_OLDAUTHTOK_PROMPT:
 *		The prompt to use when asking the applicant for an
 *		expired authentication token prior to changing it.
 *
 * See =pam_start for a description of =struct pam_conv.
 *
 * >pam_set_item
 */
