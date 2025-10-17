/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Call the conversation function
 */

int
pam_vprompt(const pam_handle_t *pamh,
	int style,
	char **resp,
	const char *fmt,
	va_list ap)
{
	char msgbuf[PAM_MAX_MSG_SIZE];
	struct pam_message msg;
	const struct pam_message *msgp;
	struct pam_response *rsp;
	const struct pam_conv *conv;
	const void *convp;
	int r;

	ENTER();
	r = pam_get_item(pamh, PAM_CONV, &convp);
	if (r != PAM_SUCCESS)
		RETURNC(r);
	conv = convp;
	if (conv == NULL || conv->conv == NULL) {
		openpam_log(PAM_LOG_ERROR, "no conversation function");
		RETURNC(PAM_SYSTEM_ERR);
	}
	vsnprintf(msgbuf, PAM_MAX_MSG_SIZE, fmt, ap);
	msg.msg_style = style;
	msg.msg = msgbuf;
	msgp = &msg;
	rsp = NULL;
	r = (conv->conv)(1, &msgp, &rsp, conv->appdata_ptr);
	*resp = rsp == NULL ? NULL : rsp->resp;
	FREE(rsp);
	RETURNC(r);
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
 * The =pam_vprompt function constructs a string from the =fmt and =ap
 * arguments using =vsnprintf, and passes it to the given PAM context's
 * conversation function.
 *
 * The =style argument specifies the type of interaction requested, and
 * must be one of the following:
 *
 *	=PAM_PROMPT_ECHO_OFF:
 *		Display the message and obtain the user's response without
 *		displaying it.
 *	=PAM_PROMPT_ECHO_ON:
 *		Display the message and obtain the user's response.
 *	=PAM_ERROR_MSG:
 *		Display the message as an error message, and do not wait
 *		for a response.
 *	=PAM_TEXT_INFO:
 *		Display the message as an informational message, and do
 *		not wait for a response.
 *
 * A pointer to the response, or =NULL if the conversation function did
 * not return one, is stored in the location pointed to by the =resp
 * argument.
 *
 * The message and response should not exceed =PAM_MAX_MSG_SIZE or
 * =PAM_MAX_RESP_SIZE, respectively.
 * If they do, they may be truncated.
 *
 * >pam_error
 * >pam_info
 * >pam_prompt
 * >pam_verror
 * >pam_vinfo
 */
