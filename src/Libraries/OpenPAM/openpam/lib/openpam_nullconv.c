/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include <sys/types.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Null conversation function
 */

int
openpam_nullconv(int n,
	 const struct pam_message **msg,
	 struct pam_response **resp,
	 void *data)
{

	ENTER();
	(void)n;
	(void)msg;
	(void)resp;
	(void)data;
	RETURNC(PAM_CONV_ERR);
}

/*
 * Error codes:
 *
 *	PAM_CONV_ERR
 */

/**
 * The =openpam_nullconv function is a null conversation function suitable
 * for applications that want to use PAM but don't support interactive
 * dialog with the user.
 * Such applications should set =PAM_AUTHTOK to whatever authentication
 * token they've obtained on their own before calling =pam_authenticate
 * and / or =pam_chauthtok, and their PAM configuration should specify the
 * ;use_first_pass option for all modules that require access to the
 * authentication token, to make sure they use =PAM_AUTHTOK rather than
 * try to query the user.
 *
 * >openpam_ttyconv
 * >pam_prompt
 * >pam_set_item
 * >pam_vprompt
 */
