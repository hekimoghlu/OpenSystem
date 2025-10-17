/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <security/pam_appl.h>
#include <security/openpam.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Display an error message
 */

int
pam_error(const pam_handle_t *pamh,
	const char *fmt,
	...)
{
	va_list ap;
	char *rsp;
	int r;

	va_start(ap, fmt);
	r = pam_vprompt(pamh, PAM_ERROR_MSG, &rsp, fmt, ap);
	va_end(ap);
	FREE(rsp); /* ignore response */
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
 * The =pam_error function displays an error message through the
 * intermediary of the given PAM context's conversation function.
 *
 * >pam_info
 * >pam_prompt
 * >pam_verror
 */
