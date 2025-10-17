/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>
#include <mail_error.h>
#include <ehlo_mask.h>

/* Application-specific. */

#include "smtpd.h"
#include "smtpd_token.h"
#include "smtpd_chat.h"
#include "smtpd_sasl_proto.h"
#include "smtpd_sasl_glue.h"

#ifdef USE_SASL_AUTH

/* smtpd_sasl_auth_cmd - process AUTH command */

int     smtpd_sasl_auth_cmd(SMTPD_STATE *state, int argc, SMTPD_TOKEN *argv)
{
    char   *auth_mechanism;
    char   *initial_response;
    const char *err;

    if (var_helo_required && state->helo_name == 0) {
	state->error_mask |= MAIL_ERROR_POLICY;
	smtpd_chat_reply(state, "503 5.5.1 Error: send HELO/EHLO first");
	return (-1);
    }
    if (SMTPD_STAND_ALONE(state) || !smtpd_sasl_is_active(state)
	|| (state->ehlo_discard_mask & EHLO_MASK_AUTH)) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	smtpd_chat_reply(state, "503 5.5.1 Error: authentication not enabled");
	return (-1);
    }
    if (SMTPD_IN_MAIL_TRANSACTION(state)) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	smtpd_chat_reply(state, "503 5.5.1 Error: MAIL transaction in progress");
	return (-1);
    }
    if (state->milters != 0 && (err = milter_other_event(state->milters)) != 0) {
	if (err[0] == '5') {
	    state->error_mask |= MAIL_ERROR_POLICY;
	    smtpd_chat_reply(state, "%s", err);
	    return (-1);
	}
	/* Sendmail compatibility: map 4xx into 454. */
	else if (err[0] == '4') {
	    state->error_mask |= MAIL_ERROR_POLICY;
	    smtpd_chat_reply(state, "454 4.3.0 Try again later");
	    return (-1);
	}
    }
#ifdef USE_TLS
    if (var_smtpd_tls_auth_only && !state->tls_context) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	/* RFC 4954, Section 4. */
	smtpd_chat_reply(state, "504 5.5.4 Encryption required for requested authentication mechanism");
	return (-1);
    }
#endif
    if (state->sasl_username) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	smtpd_chat_reply(state, "503 5.5.1 Error: already authenticated");
	return (-1);
    }
    if (argc < 2 || argc > 3) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	smtpd_chat_reply(state, "501 5.5.4 Syntax: AUTH mechanism");
	return (-1);
    }
    /* Don't reuse the SASL handle after authentication failure. */
#ifndef XSASL_TYPE_CYRUS
#define XSASL_TYPE_CYRUS	"cyrus"
#endif
    if (state->flags & SMTPD_FLAG_AUTH_USED) {
	smtpd_sasl_deactivate(state);
#ifdef USE_TLS
	if (state->tls_context != 0)
	    smtpd_sasl_activate(state, VAR_SMTPD_SASL_TLS_OPTS,
				var_smtpd_sasl_tls_opts);
	else
#endif
	    smtpd_sasl_activate(state, VAR_SMTPD_SASL_OPTS,
				var_smtpd_sasl_opts);
    } else if (strcmp(var_smtpd_sasl_type, XSASL_TYPE_CYRUS) == 0) {
	state->flags |= SMTPD_FLAG_AUTH_USED;
    }

    /*
     * All authentication failures shall be logged. The 5xx reply code from
     * the SASL authentication routine triggers tar-pit delays, which help to
     * slow down password guessing attacks.
     */
    auth_mechanism = argv[1].strval;
    initial_response = (argc == 3 ? argv[2].strval : 0);
    return (smtpd_sasl_authenticate(state, auth_mechanism, initial_response));
}

/* smtpd_sasl_mail_opt - SASL-specific MAIL FROM option */

char   *smtpd_sasl_mail_opt(SMTPD_STATE *state, const char *addr)
{

    /*
     * Do not store raw RFC2554 protocol data.
     */
#if 0
    if (state->sasl_username == 0) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	return ("503 5.5.4 Error: send AUTH command first");
    }
#endif
    if (state->sasl_sender != 0) {
	state->error_mask |= MAIL_ERROR_PROTOCOL;
	return ("503 5.5.4 Error: multiple AUTH= options");
    }
    if (strcmp(addr, "<>") != 0) {
	state->sasl_sender = mystrdup(addr);
	printable(state->sasl_sender, '?');
    }
    return (0);
}

/* smtpd_sasl_mail_reset - SASL-specific MAIL FROM cleanup */

void    smtpd_sasl_mail_reset(SMTPD_STATE *state)
{
    if (state->sasl_sender) {
	myfree(state->sasl_sender);
	state->sasl_sender = 0;
    }
}

/* permit_sasl_auth - OK for authenticated connection */

int     permit_sasl_auth(SMTPD_STATE *state, int ifyes, int ifnot)
{
    if (state->sasl_method && strcasecmp(state->sasl_method, "anonymous"))
	return (ifyes);
    return (ifnot);
}

#endif
