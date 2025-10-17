/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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

#ifdef HAVE_OSF_SIA
#include <sia.h>
#include <siad.h>
#include <pwd.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>

#include "ssh.h"
#include "ssh_api.h"
#include "hostfile.h"
#include "auth.h"
#include "auth-sia.h"
#include "log.h"
#include "servconf.h"
#include "canohost.h"
#include "uidswap.h"

extern ServerOptions options;
extern int saved_argc;
extern char **saved_argv;

int
sys_auth_passwd(struct ssh *ssh, const char *pass)
{
	int ret;
	SIAENTITY *ent = NULL;
	const char *host;
	Authctxt *authctxt = ssh->authctxt;

	host = get_canonical_hostname(options.use_dns);

	if (!authctxt->user || pass == NULL || pass[0] == '\0')
		return (0);

	if (sia_ses_init(&ent, saved_argc, saved_argv, host, authctxt->user,
	    NULL, 0, NULL) != SIASUCCESS)
		return (0);

	if ((ret = sia_ses_authent(NULL, pass, ent)) != SIASUCCESS) {
		error("Couldn't authenticate %s from %s",
		    authctxt->user, host);
		if (ret & SIASTOP)
			sia_ses_release(&ent);

		return (0);
	}

	sia_ses_release(&ent);

	return (1);
}

void
session_setup_sia(struct passwd *pw, char *tty)
{
	SIAENTITY *ent = NULL;
	const char *host;

	host = get_canonical_hostname(options.use_dns);

	if (sia_ses_init(&ent, saved_argc, saved_argv, host, pw->pw_name,
	    tty, 0, NULL) != SIASUCCESS)
		fatal("sia_ses_init failed");

	if (sia_make_entity_pwd(pw, ent) != SIASUCCESS) {
		sia_ses_release(&ent);
		fatal("sia_make_entity_pwd failed");
	}

	ent->authtype = SIA_A_NONE;
	if (sia_ses_estab(sia_collect_trm, ent) != SIASUCCESS)
		fatal("Couldn't establish session for %s from %s",
		    pw->pw_name, host);

	if (sia_ses_launch(sia_collect_trm, ent) != SIASUCCESS)
		fatal("Couldn't launch session for %s from %s",
		    pw->pw_name, host);

	sia_ses_release(&ent);

	setuid(0);
	permanently_set_uid(pw);
}

#endif /* HAVE_OSF_SIA */
