/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <stdlib.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>

/* XSASL library. */

#include <xsasl.h>

/* Application-specific. */

#include "smtpd.h"
#include "smtpd_sasl_glue.h"
#include "smtpd_chat.h"

#ifdef USE_SASL_AUTH

/*
 * Silly little macros.
 */
#define STR(s)	vstring_str(s)

 /*
  * SASL server implementation handle.
  */
static XSASL_SERVER_IMPL *smtpd_sasl_impl;

/* smtpd_sasl_initialize - per-process initialization */

void    smtpd_sasl_initialize(void)
{

    /*
     * Sanity check.
     */
    if (smtpd_sasl_impl)
	msg_panic("smtpd_sasl_initialize: repeated call");

    /*
     * Initialize the SASL library.
     */
    if ((smtpd_sasl_impl = xsasl_server_init(var_smtpd_sasl_type,
					     var_smtpd_sasl_path)) == 0)
	msg_fatal("SASL per-process initialization failed");

}

/* smtpd_sasl_activate - per-connection initialization */

void    smtpd_sasl_activate(SMTPD_STATE *state, const char *sasl_opts_name,
			            const char *sasl_opts_val)
{
    const char *mechanism_list;
    XSASL_SERVER_CREATE_ARGS create_args;
    int     tls_flag;

    /*
     * Sanity check.
     */
    if (smtpd_sasl_is_active(state))
	msg_panic("smtpd_sasl_activate: already active");

    /*
     * Initialize SASL-specific state variables. Use long-lived storage for
     * base 64 conversion results, rather than local variables, to avoid
     * memory leaks when a read or write routine returns abnormally after
     * timeout or I/O error.
     */
    state->sasl_reply = vstring_alloc(20);
    state->sasl_mechanism_list = 0;

    /*
     * Set up a new server context for this connection.
     */
#ifdef USE_TLS
    tls_flag = state->tls_context != 0;
#else
    tls_flag = 0;
#endif
#define ADDR_OR_EMPTY(addr, unknown) (strcmp(addr, unknown) ? addr : "")
#define REALM_OR_NULL(realm) (*(realm) ? (realm) : (char *) 0)

    if ((state->sasl_server =
	 XSASL_SERVER_CREATE(smtpd_sasl_impl, &create_args,
			     stream = state->client,
			     addr_family = state->addr_family,
			     server_addr = ADDR_OR_EMPTY(state->dest_addr,
						       SERVER_ADDR_UNKNOWN),
			     server_port = ADDR_OR_EMPTY(state->dest_port,
						       SERVER_PORT_UNKNOWN),
			     client_addr = ADDR_OR_EMPTY(state->addr,
						       CLIENT_ADDR_UNKNOWN),
			     client_port = ADDR_OR_EMPTY(state->port,
						       CLIENT_PORT_UNKNOWN),
			     service = var_smtpd_sasl_service,
			   user_realm = REALM_OR_NULL(var_smtpd_sasl_realm),
			     security_options = sasl_opts_val,
			     tls_flag = tls_flag)) == 0)
	msg_fatal("SASL per-connection initialization failed");

    /*
     * Get the list of authentication mechanisms.
     */
    if ((mechanism_list =
	 xsasl_server_get_mechanism_list(state->sasl_server)) == 0)
	msg_fatal("no SASL authentication mechanisms");
    state->sasl_mechanism_list = mystrdup(mechanism_list);
}

/* smtpd_sasl_state_init - initialize state to allow extern authentication. */

void    smtpd_sasl_state_init(SMTPD_STATE *state)
{
    /* Initialization to support external authentication (e.g., XCLIENT). */
    state->sasl_username = 0;
    state->sasl_method = 0;
    state->sasl_sender = 0;
}

/* smtpd_sasl_deactivate - per-connection cleanup */

void    smtpd_sasl_deactivate(SMTPD_STATE *state)
{
    if (state->sasl_reply) {
	vstring_free(state->sasl_reply);
	state->sasl_reply = 0;
    }
    if (state->sasl_mechanism_list) {
	myfree(state->sasl_mechanism_list);
	state->sasl_mechanism_list = 0;
    }
    if (state->sasl_username) {
	myfree(state->sasl_username);
	state->sasl_username = 0;
    }
    if (state->sasl_method) {
	myfree(state->sasl_method);
	state->sasl_method = 0;
    }
    if (state->sasl_sender) {
	myfree(state->sasl_sender);
	state->sasl_sender = 0;
    }
    if (state->sasl_server) {
	xsasl_server_free(state->sasl_server);
	state->sasl_server = 0;
    }
}

/* smtpd_sasl_authenticate - per-session authentication */

int     smtpd_sasl_authenticate(SMTPD_STATE *state,
				        const char *sasl_method,
				        const char *init_response)
{
    int     status;
    const char *sasl_username;

    /*
     * SASL authentication protocol start-up. Process any initial client
     * response that was sent along in the AUTH command.
     */
    for (status = xsasl_server_first(state->sasl_server, sasl_method,
				     init_response, state->sasl_reply);
	 status == XSASL_AUTH_MORE;
	 status = xsasl_server_next(state->sasl_server, STR(state->buffer),
				    state->sasl_reply)) {

	/*
	 * Send a server challenge.
	 */
	smtpd_chat_reply(state, "334 %s", STR(state->sasl_reply));

	/*
	 * Receive the client response. "*" means that the client gives up.
	 * XXX For now we ignore the fact that an excessively long response
	 * will be chopped into multiple responses. To handle such responses,
	 * we need to change smtpd_chat_query() so that it returns an error
	 * indication.
	 */
	smtpd_chat_query(state);
	if (strcmp(STR(state->buffer), "*") == 0) {
	    msg_warn("%s: SASL %s authentication aborted",
		     state->namaddr, sasl_method);
	    smtpd_chat_reply(state, "501 5.7.0 Authentication aborted");
	    return (-1);
	}
    }
    if (status != XSASL_AUTH_DONE) {
	msg_warn("%s: SASL %s authentication failed: %s",
		 state->namaddr, sasl_method,
		 STR(state->sasl_reply));
	/* RFC 4954 Section 6. */
	if (status == XSASL_AUTH_TEMP)
	    smtpd_chat_reply(state, "454 4.7.0 Temporary authentication failure: %s",
			     STR(state->sasl_reply));
	else
	    smtpd_chat_reply(state, "535 5.7.8 Error: authentication failed: %s",
			     STR(state->sasl_reply));
	return (-1);
    }
    /* RFC 4954 Section 6. */
    smtpd_chat_reply(state, "235 2.7.0 Authentication successful");
    if ((sasl_username = xsasl_server_get_username(state->sasl_server)) == 0)
	msg_panic("cannot look up the authenticated SASL username");
    state->sasl_username = mystrdup(sasl_username);
    printable(state->sasl_username, '?');
    state->sasl_method = mystrdup(sasl_method);
    printable(state->sasl_method, '?');

    return (0);
}

/* smtpd_sasl_logout - clean up after smtpd_sasl_authenticate */

void    smtpd_sasl_logout(SMTPD_STATE *state)
{
    if (state->sasl_username) {
	myfree(state->sasl_username);
	state->sasl_username = 0;
    }
    if (state->sasl_method) {
	myfree(state->sasl_method);
	state->sasl_method = 0;
    }
}

/* smtpd_sasl_login - set login information */

void    smtpd_sasl_login(SMTPD_STATE *state, const char *sasl_username,
			         const char *sasl_method)
{
    if (state->sasl_username)
	myfree(state->sasl_username);
    state->sasl_username = mystrdup(sasl_username);
    if (state->sasl_method)
	myfree(state->sasl_method);
    state->sasl_method = mystrdup(sasl_method);
}

#endif
