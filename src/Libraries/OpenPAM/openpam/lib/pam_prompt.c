/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#include <stdarg.h>

#include <security/pam_appl.h>
#include <security/openpam.h>

/*
 * OpenPAM extension
 *
 * Call the conversation function
 */

int
pam_prompt(const pam_handle_t *pamh,
	int style,
	char **resp,
	const char *fmt,
	...)
{
	va_list ap;
	int r;

	va_start(ap, fmt);
	r = pam_vprompt(pamh, style, resp, fmt, ap);
	va_end(ap);
	return (r);
}

/*
 * Error codes:
 *
 *     !PAM_SYMBOL_ERR
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 *	PAM_CONV_ERR
 */

/**
 * The =pam_prompt function constructs a message from the specified format
 * string and arguments and passes it to the given PAM context's
 * conversation function.
 *
 * A pointer to the response, or =NULL if the conversation function did
 * not return one, is stored in the location pointed to by the =resp
 * argument.
 *
 * See =pam_vprompt for further details.
 *
 * >pam_error
 * >pam_info
 * >pam_vprompt
 */
