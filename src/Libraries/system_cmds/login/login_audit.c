/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifdef USE_BSM_AUDIT

#include <sys/types.h>

#include <bsm/libbsm.h>
#include <bsm/audit_uevents.h>
#include <bsm/audit_session.h>

#ifdef __APPLE__
#if __has_include(<EndpointSecuritySystem/ESSubmitSPI.h>)
#include <EndpointSecuritySystem/ESSubmitSPI.h>
#endif /* __has_include */
#include <SoftLinking/WeakLinking.h>
WEAK_LINK_FORCE_IMPORT(ess_notify_login_login);
WEAK_LINK_FORCE_IMPORT(ess_notify_login_logout);
#endif /* __APPLE__ */

#include <err.h>
#include <errno.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

#include "login.h"

/*
 * Audit data
 */
au_tid_addr_t tid;

// rdar://87888376 (Adopt EndpointSecurity event submission SPI - ess_notify_login_login/logout)
// The old audit APIs are deprecated, ignore those errors since we already adopted libEndpointSecuritySystem
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/*
 * The following tokens are included in the audit record for a successful
 * login: header, subject, return.
 */
void
au_login_success(int fflag)
{
	token_t *tok;
	int aufd;
	auditinfo_addr_t auinfo;
	uid_t uid = pwd->pw_uid;
	gid_t gid = pwd->pw_gid;
	pid_t pid = getpid();
	int au_cond;

	/* If we are not auditing, don't cut an audit record; just return. */
 	if (auditon(A_GETCOND, &au_cond, sizeof(au_cond)) < 0) {
		if (errno == ENOSYS)
			return;
		errx(1, "could not determine audit condition");
	}

	/* Initialize with the current audit info. */
	if (getaudit_addr(&auinfo, sizeof(auinfo)) < 0) {
		err(1, "getaudit_addr");
	}
	auinfo.ai_auid = pwd->pw_uid;
	memcpy(&auinfo.ai_termid, &tid, sizeof(auinfo.ai_termid));

	/* Do the SessionCreate() equivalent. */
	if (!fflag) {
		auinfo.ai_asid = AU_ASSIGN_ASID;
		auinfo.ai_flags |= AU_SESSION_FLAG_HAS_TTY;
		auinfo.ai_flags |= AU_SESSION_FLAG_HAS_AUTHENTICATED;
	}

	if (au_cond != AUC_NOAUDIT) {
		/* Compute and set the user's preselection mask. */
		if (au_user_mask(pwd->pw_name, &auinfo.ai_mask) < 0) {
			errx(1, "could not set audit mask\n");
		}
	}

	if (setaudit_addr(&auinfo, sizeof(auinfo)) < 0)
		err(1, "setaudit_addr failed");

	char *session = NULL;
	asprintf(&session, "%x", auinfo.ai_asid);
	if (NULL == session) {
		errx(1, "asprintf failed");
	}
	setenv("SECURITYSESSIONID", session, 1);
	free(session);

#ifdef __APPLE__
	/*
	 * Emit ES event after setting up the audit session, so the audit
	 * session ID and audit user ID in the event represent the newly
	 * created session for this login.
	 */
	if (ess_notify_login_login != NULL)
		ess_notify_login_login(true, NULL, pwd->pw_name, &uid);
#endif /* __APPLE__ */

	/* If we are not auditing, don't cut an audit record; just return. */
	if (au_cond == AUC_NOAUDIT)
		return;

	if ((aufd = au_open()) == -1)
		errx(1, "audit error: au_open() failed");

	if ((tok = au_to_subject32_ex(uid, geteuid(), getegid(), uid, gid, pid,
	    pid, &tid)) == NULL)
		errx(1, "audit error: au_to_subject32() failed");
	au_write(aufd, tok);

	if ((tok = au_to_return32(0, 0)) == NULL)
		errx(1, "audit error: au_to_return32() failed");
	au_write(aufd, tok);

	if (au_close(aufd, 1, AUE_login) == -1)
		errx(1, "audit record was not committed.");
}

/*
 * The following tokens are included in the audit record for failed
 * login attempts: header, subject, text, return.
 */
void
au_login_fail(const char *errmsg, int na, const char *username, int fflag)
{
	token_t *tok;
	int aufd;
	int au_cond;
	uid_t uid;
	gid_t gid;
	pid_t pid = getpid();

#ifdef __APPLE__
	if (!fflag &&
	    ess_notify_login_login != NULL) {
		if (!na)
			ess_notify_login_login(false, errmsg, pwd->pw_name, &pwd->pw_uid);
		else if (username != NULL)
			ess_notify_login_login(false, errmsg, username, NULL);
	}
#endif /* __APPLE__ */

	/* If we are not auditing, don't cut an audit record; just return. */
 	if (auditon(A_GETCOND, &au_cond, sizeof(au_cond)) < 0) {
		if (errno == ENOSYS)
			return;
		errx(1, "could not determine audit condition");
	}
	if (au_cond == AUC_NOAUDIT)
		return;

	if ((aufd = au_open()) == -1)
		errx(1, "audit error: au_open() failed");

	if (na) {
		/*
		 * Non attributable event.  Assuming that login is not called
		 * within a user's session => auid,asid == -1.
		 */
		if ((tok = au_to_subject32_ex(-1, geteuid(), getegid(), -1, -1,
		    pid, -1, &tid)) == NULL)
			errx(1, "audit error: au_to_subject32() failed");
	} else {
		/* We know the subject -- so use its value instead. */
		uid = pwd->pw_uid;
		gid = pwd->pw_gid;
		if ((tok = au_to_subject32_ex(uid, geteuid(), getegid(), uid,
		    gid, pid, pid, &tid)) == NULL)
			errx(1, "audit error: au_to_subject32() failed");
	}
	au_write(aufd, tok);

	/* Include the error message. */
	if ((tok = au_to_text(errmsg)) == NULL)
		errx(1, "audit error: au_to_text() failed");
	au_write(aufd, tok);

	if ((tok = au_to_return32(1, errno)) == NULL)
		errx(1, "audit error: au_to_return32() failed");
	au_write(aufd, tok);

	if (au_close(aufd, 1, AUE_login) == -1)
		errx(1, "audit error: au_close() was not committed");
}

/*
 * The following tokens are included in the audit record for a logout:
 * header, subject, return.
 */
void
audit_logout(int fflag)
{
	token_t *tok;
	int aufd;
	uid_t uid = pwd->pw_uid;
	gid_t gid = pwd->pw_gid;
	pid_t pid = getpid();
	int au_cond;

#ifdef __APPLE__
	if (!fflag &&
	    ess_notify_login_logout != NULL)
		ess_notify_login_logout(pwd->pw_name, uid);
#endif /* __APPLE__ */

	/* If we are not auditing, don't cut an audit record; just return. */
	if (auditon(A_GETCOND, &au_cond, sizeof(au_cond)) < 0) {
		if (errno == ENOSYS)
			return;
		errx(1, "could not determine audit condition");
	}
	if (au_cond == AUC_NOAUDIT)
		return;

	if ((aufd = au_open()) == -1)
		errx(1, "audit error: au_open() failed");

	/* The subject that is created (euid, egid of the current process). */
	if ((tok = au_to_subject32_ex(uid, geteuid(), getegid(), uid, gid, pid,
	    pid, &tid)) == NULL)
		errx(1, "audit error: au_to_subject32() failed");
	au_write(aufd, tok);

	if ((tok = au_to_return32(0, 0)) == NULL)
		errx(1, "audit error: au_to_return32() failed");
	au_write(aufd, tok);

	if (au_close(aufd, 1, AUE_logout) == -1)
		errx(1, "audit record was not committed.");
}

#pragma clang diagnostic pop

#endif /* USE_BSM_AUDIT */
