/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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

/* Utility library. */

#include <split_at.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <quote_821_local.h>

/* Milter library. */

#include <milter.h>

/* Application-specific. */

#include <smtpd.h>
#include <smtpd_sasl_glue.h>
#include <smtpd_resolve.h>
#include <smtpd_milter.h>

 /*
  * SLMs.
  */
#define STR(x)	vstring_str(x)

/* smtpd_milter_eval - evaluate milter macro */

const char *smtpd_milter_eval(const char *name, void *ptr)
{
    SMTPD_STATE *state = (SMTPD_STATE *) ptr;
    const RESOLVE_REPLY *reply;
    char   *cp;

    /*
     * On-the-fly initialization.
     */
    if (state->expand_buf == 0)
	state->expand_buf = vstring_alloc(10);

    /*
     * System macros.
     */
    if (strcmp(name, S8_MAC_DAEMON_NAME) == 0)
	return (var_milt_daemon_name);
    if (strcmp(name, S8_MAC_V) == 0)
	return (var_milt_v);

    /*
     * Connect macros.
     */
    if (strcmp(name, S8_MAC__) == 0) {
	vstring_sprintf(state->expand_buf, "%s [%s]",
			state->reverse_name, state->addr);
	if (strcasecmp_utf8(state->name, state->reverse_name) != 0)
	    vstring_strcat(state->expand_buf, " (may be forged)");
	return (STR(state->expand_buf));
    }
    if (strcmp(name, S8_MAC_J) == 0)
	return (var_myhostname);
    if (strcmp(name, S8_MAC_CLIENT_ADDR) == 0)
	return (state->rfc_addr);
    if (strcmp(name, S8_MAC_CLIENT_PORT) == 0)
	return (strcmp(state->port, CLIENT_PORT_UNKNOWN) ? state->port : "0");
    if (strcmp(name, S8_MAC_CLIENT_CONN) == 0) {
	vstring_sprintf(state->expand_buf, "%d", state->conn_count);
	return (STR(state->expand_buf));
    }
    if (strcmp(name, S8_MAC_CLIENT_NAME) == 0)
	return (state->name);
    if (strcmp(name, S8_MAC_CLIENT_PTR) == 0)
	return (state->reverse_name);
    if (strcmp(name, S8_MAC_CLIENT_RES) == 0)
	return (state->name_status == SMTPD_PEER_CODE_OK ? "OK" :
		state->name_status == SMTPD_PEER_CODE_FORGED ? "FORGED" :
	      state->name_status == SMTPD_PEER_CODE_TEMP ? "TEMP" : "FAIL");

    if (strcmp(name, S8_MAC_DAEMON_ADDR) == 0)
	return (state->dest_addr);
    if (strcmp(name, S8_MAC_DAEMON_PORT) == 0)
	return (state->dest_port);

    /*
     * HELO macros.
     */
#ifdef USE_TLS
#define IF_ENCRYPTED(x) (state->tls_context ? (x) : 0)
#define IF_TRUSTED(x) (TLS_CERT_IS_TRUSTED(state->tls_context) ? (x) : 0)

    if (strcmp(name, S8_MAC_TLS_VERSION) == 0)
	return (IF_ENCRYPTED(state->tls_context->protocol));
    if (strcmp(name, S8_MAC_CIPHER) == 0)
	return (IF_ENCRYPTED(state->tls_context->cipher_name));
    if (strcmp(name, S8_MAC_CIPHER_BITS) == 0) {
	if (state->tls_context == 0)
	    return (0);
	vstring_sprintf(state->expand_buf, "%d",
			IF_ENCRYPTED(state->tls_context->cipher_usebits));
	return (STR(state->expand_buf));
    }
    if (strcmp(name, S8_MAC_CERT_SUBJECT) == 0)
	return (IF_TRUSTED(state->tls_context->peer_CN));
    if (strcmp(name, S8_MAC_CERT_ISSUER) == 0)
	return (IF_TRUSTED(state->tls_context->issuer_CN));
#endif

    /*
     * MAIL FROM macros.
     */
#define IF_SASL_ENABLED(s) ((s) ? (s) : 0)

    if (strcmp(name, S8_MAC_I) == 0)
	return (state->queue_id);
#ifdef USE_SASL_AUTH
    if (strcmp(name, S8_MAC_AUTH_TYPE) == 0)
	return (IF_SASL_ENABLED(state->sasl_method));
    if (strcmp(name, S8_MAC_AUTH_AUTHEN) == 0)
	return (IF_SASL_ENABLED(state->sasl_username));
    if (strcmp(name, S8_MAC_AUTH_AUTHOR) == 0)
	return (IF_SASL_ENABLED(state->sasl_sender));
#endif
    if (strcmp(name, S8_MAC_MAIL_ADDR) == 0) {
	if (state->sender == 0)
	    return (0);
	if (state->sender[0] == 0)
	    return ("");
	reply = smtpd_resolve_addr(state->recipient, state->sender);
	/* Sendmail 8.13 does not externalize the null string. */
	if (STR(reply->recipient)[0])
	    quote_821_local(state->expand_buf, STR(reply->recipient));
	else
	    vstring_strcpy(state->expand_buf, STR(reply->recipient));
	return (STR(state->expand_buf));
    }
    if (strcmp(name, S8_MAC_MAIL_HOST) == 0) {
	if (state->sender == 0)
	    return (0);
	reply = smtpd_resolve_addr(state->recipient, state->sender);
	return (STR(reply->nexthop));
    }
    if (strcmp(name, S8_MAC_MAIL_MAILER) == 0) {
	if (state->sender == 0)
	    return (0);
	reply = smtpd_resolve_addr(state->recipient, state->sender);
	return (STR(reply->transport));
    }

    /*
     * RCPT TO macros.
     */
    if (strcmp(name, S8_MAC_RCPT_ADDR) == 0) {
	if (state->recipient == 0)
	    return (0);
	if (state->recipient[0] == 0)
	    return ("");
	if (state->milter_reject_text) {
	    /* 554 5.7.1 <user@example.com>: Relay access denied */
	    vstring_strcpy(state->expand_buf, state->milter_reject_text + 4);
	    cp = split_at(STR(state->expand_buf), ' ');
	    return (cp ? split_at(cp, ' ') : cp);
	}
	reply = smtpd_resolve_addr(state->sender, state->recipient);
	/* Sendmail 8.13 does not externalize the null string. */
	if (STR(reply->recipient)[0])
	    quote_821_local(state->expand_buf, STR(reply->recipient));
	else
	    vstring_strcpy(state->expand_buf, STR(reply->recipient));
	return (STR(state->expand_buf));
    }
    if (strcmp(name, S8_MAC_RCPT_HOST) == 0) {
	if (state->recipient == 0)
	    return (0);
	if (state->milter_reject_text) {
	    /* 554 5.7.1 <user@example.com>: Relay access denied */
	    vstring_strcpy(state->expand_buf, state->milter_reject_text + 4);
	    (void) split_at(STR(state->expand_buf), ' ');
	    return (STR(state->expand_buf));
	}
	reply = smtpd_resolve_addr(state->sender, state->recipient);
	return (STR(reply->nexthop));
    }
    if (strcmp(name, S8_MAC_RCPT_MAILER) == 0) {
	if (state->recipient == 0)
	    return (0);
	if (state->milter_reject_text)
	    return (S8_RCPT_MAILER_ERROR);
	reply = smtpd_resolve_addr(state->sender, state->recipient);
	return (STR(reply->transport));
    }
    return (0);
}
