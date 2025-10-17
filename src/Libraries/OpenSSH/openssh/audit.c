/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "includes.h"

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#ifdef SSH_AUDIT_EVENTS

#include "audit.h"
#include "log.h"
#include "hostfile.h"
#include "auth.h"

/*
 * Care must be taken when using this since it WILL NOT be initialized when
 * audit_connection_from() is called and MAY NOT be initialized when
 * audit_event(CONNECTION_ABANDON) is called.  Test for NULL before using.
 */
extern Authctxt *the_authctxt;

/* Maybe add the audit class to struct Authmethod? */
ssh_audit_event_t
audit_classify_auth(const char *method)
{
	if (strcmp(method, "none") == 0)
		return SSH_AUTH_FAIL_NONE;
	else if (strcmp(method, "password") == 0)
		return SSH_AUTH_FAIL_PASSWD;
	else if (strcmp(method, "publickey") == 0 ||
	    strcmp(method, "rsa") == 0)
		return SSH_AUTH_FAIL_PUBKEY;
	else if (strncmp(method, "keyboard-interactive", 20) == 0 ||
	    strcmp(method, "challenge-response") == 0)
		return SSH_AUTH_FAIL_KBDINT;
	else if (strcmp(method, "hostbased") == 0 ||
	    strcmp(method, "rhosts-rsa") == 0)
		return SSH_AUTH_FAIL_HOSTBASED;
	else if (strcmp(method, "gssapi-with-mic") == 0)
		return SSH_AUTH_FAIL_GSSAPI;
	else
		return SSH_AUDIT_UNKNOWN;
}

/* helper to return supplied username */
const char *
audit_username(void)
{
	static const char unknownuser[] = "(unknown user)";
	static const char invaliduser[] = "(invalid user)";

	if (the_authctxt == NULL || the_authctxt->user == NULL)
		return (unknownuser);
	if (!the_authctxt->valid)
		return (invaliduser);
	return (the_authctxt->user);
}

const char *
audit_event_lookup(ssh_audit_event_t ev)
{
	int i;
	static struct event_lookup_struct {
		ssh_audit_event_t event;
		const char *name;
	} event_lookup[] = {
		{SSH_LOGIN_EXCEED_MAXTRIES,	"LOGIN_EXCEED_MAXTRIES"},
		{SSH_LOGIN_ROOT_DENIED,		"LOGIN_ROOT_DENIED"},
		{SSH_AUTH_SUCCESS,		"AUTH_SUCCESS"},
		{SSH_AUTH_FAIL_NONE,		"AUTH_FAIL_NONE"},
		{SSH_AUTH_FAIL_PASSWD,		"AUTH_FAIL_PASSWD"},
		{SSH_AUTH_FAIL_KBDINT,		"AUTH_FAIL_KBDINT"},
		{SSH_AUTH_FAIL_PUBKEY,		"AUTH_FAIL_PUBKEY"},
		{SSH_AUTH_FAIL_HOSTBASED,	"AUTH_FAIL_HOSTBASED"},
		{SSH_AUTH_FAIL_GSSAPI,		"AUTH_FAIL_GSSAPI"},
		{SSH_INVALID_USER,		"INVALID_USER"},
		{SSH_NOLOGIN,			"NOLOGIN"},
		{SSH_CONNECTION_CLOSE,		"CONNECTION_CLOSE"},
		{SSH_CONNECTION_ABANDON,	"CONNECTION_ABANDON"},
		{SSH_AUDIT_UNKNOWN,		"AUDIT_UNKNOWN"}
	};

	for (i = 0; event_lookup[i].event != SSH_AUDIT_UNKNOWN; i++)
		if (event_lookup[i].event == ev)
			break;
	return(event_lookup[i].name);
}

# ifndef CUSTOM_SSH_AUDIT_EVENTS
/*
 * Null implementations of audit functions.
 * These get used if SSH_AUDIT_EVENTS is defined but no audit module is enabled.
 */

/*
 * Called after a connection has been accepted but before any authentication
 * has been attempted.
 */
void
audit_connection_from(const char *host, int port)
{
	debug("audit connection from %s port %d euid %d", host, port,
	    (int)geteuid());
}

/*
 * Called when various events occur (see audit.h for a list of possible
 * events and what they mean).
 */
void
audit_event(struct ssh *ssh, ssh_audit_event_t event)
{
	debug("audit event euid %d user %s event %d (%s)", geteuid(),
	    audit_username(), event, audit_event_lookup(event));
}

/*
 * Called when a user session is started.  Argument is the tty allocated to
 * the session, or NULL if no tty was allocated.
 *
 * Note that this may be called multiple times if multiple sessions are used
 * within a single connection.
 */
void
audit_session_open(struct logininfo *li)
{
	const char *t = li->line ? li->line : "(no tty)";

	debug("audit session open euid %d user %s tty name %s", geteuid(),
	    audit_username(), t);
}

/*
 * Called when a user session is closed.  Argument is the tty allocated to
 * the session, or NULL if no tty was allocated.
 *
 * Note that this may be called multiple times if multiple sessions are used
 * within a single connection.
 */
void
audit_session_close(struct logininfo *li)
{
	const char *t = li->line ? li->line : "(no tty)";

	debug("audit session close euid %d user %s tty name %s", geteuid(),
	    audit_username(), t);
}

/*
 * This will be called when a user runs a non-interactive command.  Note that
 * it may be called multiple times for a single connection since SSH2 allows
 * multiple sessions within a single connection.
 */
void
audit_run_command(const char *command)
{
	debug("audit run command euid %d user %s command '%.200s'", geteuid(),
	    audit_username(), command);
}
# endif  /* !defined CUSTOM_SSH_AUDIT_EVENTS */
#endif /* SSH_AUDIT_EVENTS */
