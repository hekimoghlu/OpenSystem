/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

#if defined(USE_SHADOW) && defined(HAS_SHADOW_EXPIRE)
#include <shadow.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#include "hostfile.h"
#include "auth.h"
#include "sshbuf.h"
#include "ssherr.h"
#include "log.h"

#ifdef DAY
# undef DAY
#endif
#define DAY	(24L * 60 * 60) /* 1 day in seconds */

extern struct sshbuf *loginmsg;

/*
 * For the account and password expiration functions, we assume the expiry
 * occurs the day after the day specified.
 */

/*
 * Check if specified account is expired.  Returns 1 if account is expired,
 * 0 otherwise.
 */
int
auth_shadow_acctexpired(struct spwd *spw)
{
	time_t today;
	long long daysleft;
	int r;

	today = time(NULL) / DAY;
	daysleft = spw->sp_expire - today;
	debug3("%s: today %lld sp_expire %lld days left %lld", __func__,
	    (long long)today, (long long)spw->sp_expire, daysleft);

	if (spw->sp_expire == -1) {
		debug3("account expiration disabled");
	} else if (daysleft < 0) {
		logit("Account %.100s has expired", spw->sp_namp);
		return 1;
	} else if (daysleft <= spw->sp_warn) {
		debug3("account will expire in %lld days", daysleft);
		if ((r = sshbuf_putf(loginmsg, 
		    "Your account will expire in %lld day%s.\n", daysleft,
		    daysleft == 1 ? "" : "s")) != 0)
			fatal("%s: buffer error: %s", __func__, ssh_err(r));
	}

	return 0;
}

/*
 * Checks password expiry for platforms that use shadow passwd files.
 * Returns: 1 = password expired, 0 = password not expired
 */
int
auth_shadow_pwexpired(Authctxt *ctxt)
{
	struct spwd *spw = NULL;
	const char *user = ctxt->pw->pw_name;
	time_t today;
	int r, daysleft, disabled = 0;

	if ((spw = getspnam((char *)user)) == NULL) {
		error("Could not get shadow information for %.100s", user);
		return 0;
	}

	today = time(NULL) / DAY;
	debug3_f("today %lld sp_lstchg %lld sp_max %lld", (long long)today,
	    (long long)spw->sp_lstchg, (long long)spw->sp_max);

#if defined(__hpux) && !defined(HAVE_SECUREWARE)
	if (iscomsec()) {
		struct pr_passwd *pr;

		pr = getprpwnam((char *)user);

		/* Test for Trusted Mode expiry disabled */
		if (pr != NULL && pr->ufld.fd_min == 0 &&
		    pr->ufld.fd_lifetime == 0 && pr->ufld.fd_expire == 0 &&
		    pr->ufld.fd_pw_expire_warning == 0 &&
		    pr->ufld.fd_schange != 0)
			disabled = 1;
	}
#endif

	/* TODO: check sp_inact */
	daysleft = spw->sp_lstchg + spw->sp_max - today;
	if (disabled) {
		debug3("password expiration disabled");
	} else if (spw->sp_lstchg == 0) {
		logit("User %.100s password has expired (root forced)", user);
		return 1;
	} else if (spw->sp_max == -1) {
		debug3("password expiration disabled");
	} else if (daysleft < 0) {
		logit("User %.100s password has expired (password aged)", user);
		return 1;
	} else if (daysleft <= spw->sp_warn) {
		debug3("password will expire in %d days", daysleft);
		if ((r = sshbuf_putf(loginmsg, 
		    "Your password will expire in %d day%s.\n", daysleft,
		    daysleft == 1 ? "" : "s")) != 0)
			fatal("%s: buffer error: %s", __func__, ssh_err(r));
	}

	return 0;
}
#endif	/* USE_SHADOW && HAS_SHADOW_EXPIRE */
