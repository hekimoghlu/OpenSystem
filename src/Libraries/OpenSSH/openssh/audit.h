/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#ifndef _SSH_AUDIT_H
# define _SSH_AUDIT_H

#include "loginrec.h"

struct ssh;

enum ssh_audit_event_type {
	SSH_LOGIN_EXCEED_MAXTRIES,
	SSH_LOGIN_ROOT_DENIED,
	SSH_AUTH_SUCCESS,
	SSH_AUTH_FAIL_NONE,
	SSH_AUTH_FAIL_PASSWD,
	SSH_AUTH_FAIL_KBDINT,	/* keyboard-interactive or challenge-response */
	SSH_AUTH_FAIL_PUBKEY,	/* ssh2 pubkey or ssh1 rsa */
	SSH_AUTH_FAIL_HOSTBASED,	/* ssh2 hostbased or ssh1 rhostsrsa */
	SSH_AUTH_FAIL_GSSAPI,
	SSH_INVALID_USER,
	SSH_NOLOGIN,		/* denied by /etc/nologin, not implemented */
	SSH_CONNECTION_CLOSE,	/* closed after attempting auth or session */
	SSH_CONNECTION_ABANDON,	/* closed without completing auth */
	SSH_AUDIT_UNKNOWN
};
typedef enum ssh_audit_event_type ssh_audit_event_t;

void	audit_connection_from(const char *, int);
void	audit_event(struct ssh *, ssh_audit_event_t);
void	audit_session_open(struct logininfo *);
void	audit_session_close(struct logininfo *);
void	audit_run_command(const char *);
ssh_audit_event_t audit_classify_auth(const char *);

#endif /* _SSH_AUDIT_H */
