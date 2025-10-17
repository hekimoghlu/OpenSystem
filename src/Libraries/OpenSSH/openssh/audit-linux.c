/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#if defined(USE_LINUX_AUDIT)
#include <libaudit.h>
#include <unistd.h>
#include <string.h>

#include "log.h"
#include "audit.h"
#include "canohost.h"
#include "packet.h"

const char *audit_username(void);

int
linux_audit_record_event(int uid, const char *username, const char *hostname,
    const char *ip, const char *ttyn, int success)
{
	int audit_fd, rc, saved_errno;

	if ((audit_fd = audit_open()) < 0) {
		if (errno == EINVAL || errno == EPROTONOSUPPORT ||
		    errno == EAFNOSUPPORT)
			return 1; /* No audit support in kernel */
		else
			return 0; /* Must prevent login */
	}
	rc = audit_log_acct_message(audit_fd, AUDIT_USER_LOGIN,
	    NULL, "login", username ? username : "(unknown)",
	    username == NULL ? uid : -1, hostname, ip, ttyn, success);
	saved_errno = errno;
	close(audit_fd);

	/*
	 * Do not report error if the error is EPERM and sshd is run as non
	 * root user.
	 */
	if ((rc == -EPERM) && (geteuid() != 0))
		rc = 0;
	errno = saved_errno;

	return rc >= 0;
}

/* Below is the sshd audit API code */

void
audit_connection_from(const char *host, int port)
{
	/* not implemented */
}

void
audit_run_command(const char *command)
{
	/* not implemented */
}

void
audit_session_open(struct logininfo *li)
{
	if (linux_audit_record_event(li->uid, NULL, li->hostname, NULL,
	    li->line, 1) == 0)
		fatal("linux_audit_write_entry failed: %s", strerror(errno));
}

void
audit_session_close(struct logininfo *li)
{
	/* not implemented */
}

void
audit_event(struct ssh *ssh, ssh_audit_event_t event)
{
	switch(event) {
	case SSH_AUTH_SUCCESS:
	case SSH_CONNECTION_CLOSE:
	case SSH_NOLOGIN:
	case SSH_LOGIN_EXCEED_MAXTRIES:
	case SSH_LOGIN_ROOT_DENIED:
		break;
	case SSH_AUTH_FAIL_NONE:
	case SSH_AUTH_FAIL_PASSWD:
	case SSH_AUTH_FAIL_KBDINT:
	case SSH_AUTH_FAIL_PUBKEY:
	case SSH_AUTH_FAIL_HOSTBASED:
	case SSH_AUTH_FAIL_GSSAPI:
	case SSH_INVALID_USER:
		linux_audit_record_event(-1, audit_username(), NULL,
		    ssh_remote_ipaddr(ssh), "sshd", 0);
		break;
	default:
		debug("%s: unhandled event %d", __func__, event);
		break;
	}
}
#endif /* USE_LINUX_AUDIT */
