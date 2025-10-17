/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#ifdef HAVE_SIA_SES_INIT

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#include <signal.h>
#include <siad.h>

#include "sudoers.h"
#include "sudo_auth.h"

static char **sudo_argv;
static int sudo_argc;

int
sudo_sia_setup(struct passwd *pw, char **promptp, sudo_auth *auth)
{
    SIAENTITY *siah;
    int i;
    debug_decl(sudo_sia_setup, SUDOERS_DEBUG_AUTH);

    /* Rebuild argv for sia_ses_init() */
    sudo_argc = NewArgc + 1;
    sudo_argv = reallocarray(NULL, sudo_argc + 1, sizeof(char *));
    if (sudo_argv == NULL) {
	log_warningx(0, N_("unable to allocate memory"));
	debug_return_int(AUTH_FATAL);
    }
    sudo_argv[0] = "sudo";
    for (i = 0; i < NewArgc; i++)
	sudo_argv[i + 1] = NewArgv[i];
    sudo_argv[sudo_argc] = NULL;

    /* We don't let SIA prompt the user for input. */
    if (sia_ses_init(&siah, sudo_argc, sudo_argv, NULL, pw->pw_name, user_ttypath, 0, NULL) != SIASUCCESS) {
	log_warning(0, N_("unable to initialize SIA session"));
	debug_return_int(AUTH_FATAL);
    }

    auth->data = siah;
    debug_return_int(AUTH_SUCCESS);
}

int
sudo_sia_verify(struct passwd *pw, const char *prompt, sudo_auth *auth,
    struct sudo_conv_callback *callback)
{
    SIAENTITY *siah = auth->data;
    char *pass;
    int rc;
    debug_decl(sudo_sia_verify, SUDOERS_DEBUG_AUTH);

    if (IS_NONINTERACTIVE(auth))
        debug_return_int(AUTH_NONINTERACTIVE);

    /* Get password, return AUTH_INTR if we got ^C */
    pass = auth_getpass(prompt, SUDO_CONV_PROMPT_ECHO_OFF, callback);
    if (pass == NULL)
	debug_return_int(AUTH_INTR);

    /* Check password and zero out plaintext copy. */
    rc = sia_ses_authent(NULL, pass, siah);
    freezero(pass, strlen(pass));

    if (rc == SIASUCCESS)
	debug_return_int(AUTH_SUCCESS);
    if (ISSET(rc, SIASTOP))
	debug_return_int(AUTH_FATAL);
    debug_return_int(AUTH_FAILURE);
}

int
sudo_sia_cleanup(struct passwd *pw, sudo_auth *auth, bool force)
{
    SIAENTITY *siah = auth->data;
    debug_decl(sudo_sia_cleanup, SUDOERS_DEBUG_AUTH);

    (void) sia_ses_release(&siah);
    auth->data = NULL;
    free(sudo_argv);
    debug_return_int(AUTH_SUCCESS);
}

int
sudo_sia_begin_session(struct passwd *pw, char **user_envp[], sudo_auth *auth)
{
    SIAENTITY *siah;
    int status = AUTH_FATAL;
    debug_decl(sudo_sia_begin_session, SUDOERS_DEBUG_AUTH);

    /* Re-init sia for the target user's session. */
    if (sia_ses_init(&siah, NewArgc, NewArgv, NULL, pw->pw_name, user_ttypath, 0, NULL) != SIASUCCESS) {
	log_warning(0, N_("unable to initialize SIA session"));
	goto done;
    }

    if (sia_make_entity_pwd(pw, siah) != SIASUCCESS) {
	sudo_warn("sia_make_entity_pwd");
	goto done;
    }

    status = AUTH_FAILURE;		/* no more fatal errors. */

    siah->authtype = SIA_A_NONE;
    if (sia_ses_estab(sia_collect_trm, siah) != SIASUCCESS) {
	sudo_warn("sia_ses_estab");
	goto done;
    }

    if (sia_ses_launch(sia_collect_trm, siah) != SIASUCCESS) {
	sudo_warn("sia_ses_launch");
	goto done;
    }

    status = AUTH_SUCCESS;

done:
    (void) sia_ses_release(&siah);
    debug_return_int(status);
}

#endif /* HAVE_SIA_SES_INIT */
