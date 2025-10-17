/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

#ifdef HAVE_SECURID

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

/* Needed for SecurID v5.0 Authentication on UNIX */
#define UNIX 1
#include <acexport.h>
#include <sdacmvls.h>

#include "sudoers.h"
#include "sudo_auth.h"

/*
 * securid_init - Initialises communications with ACE server
 * Arguments in:
 *     pw - UNUSED
 *     auth - sudo authentication structure
 *
 * Results out:
 *     auth - auth->data contains pointer to new SecurID handle
 *     return code - Fatal if initialization unsuccessful, otherwise
 *                   success.
 */
int
sudo_securid_init(struct passwd *pw, sudo_auth *auth)
{
    static SDI_HANDLE sd_dat;			/* SecurID handle */
    debug_decl(sudo_securid_init, SUDOERS_DEBUG_AUTH);

    /* Only initialize once. */
    if (auth->data != NULL)
	debug_return_int(AUTH_SUCCESS);

    if (IS_NONINTERACTIVE(auth))
        debug_return_int(AUTH_NONINTERACTIVE);

    /* Start communications */
    if (AceInitialize() == SD_FALSE) {
	sudo_warnx("%s", U_("failed to initialise the ACE API library"));
	debug_return_int(AUTH_FATAL);
    }

    auth->data = (void *) &sd_dat;		/* For method-specific data */

    debug_return_int(AUTH_SUCCESS);
}

/*
 * securid_setup - Initialises a SecurID transaction and locks out other
 *     ACE servers
 *
 * Arguments in:
 *     pw - struct passwd for username
 *     promptp - UNUSED
 *     auth - sudo authentication structure for SecurID handle
 *
 * Results out:
 *     return code - Success if transaction started correctly, fatal
 *                   otherwise
 */
int
sudo_securid_setup(struct passwd *pw, char **promptp, sudo_auth *auth)
{
    SDI_HANDLE *sd = (SDI_HANDLE *) auth->data;
    int retval;
    debug_decl(sudo_securid_setup, SUDOERS_DEBUG_AUTH);

    /* Re-initialize SecurID every time. */
    if (SD_Init(sd) != ACM_OK) {
	sudo_warnx("%s", U_("unable to contact the SecurID server"));
	debug_return_int(AUTH_FATAL);
    }

    /* Lock new PIN code */
    retval = SD_Lock(*sd, pw->pw_name);

    switch (retval) {
	case ACM_OK:
		sudo_warnx("%s", U_("User ID locked for SecurID Authentication"));
		debug_return_int(AUTH_SUCCESS);

        case ACE_UNDEFINED_USERNAME:
		sudo_warnx("%s", U_("invalid username length for SecurID"));
		debug_return_int(AUTH_FATAL);

	case ACE_ERR_INVALID_HANDLE:
		sudo_warnx("%s", U_("invalid Authentication Handle for SecurID"));
		debug_return_int(AUTH_FATAL);

	case ACM_ACCESS_DENIED:
		sudo_warnx("%s", U_("SecurID communication failed"));
		debug_return_int(AUTH_FATAL);

	default:
		sudo_warnx("%s", U_("unknown SecurID error"));
		debug_return_int(AUTH_FATAL);
	}
}

/*
 * securid_verify - Authenticates user and handles ACE responses
 *
 * Arguments in:
 *     pw - struct passwd for username
 *     prompt - UNUSED
 *     auth - sudo authentication structure for SecurID handle
 *
 * Results out:
 *     return code - Success on successful authentication, failure on
 *                   incorrect authentication, fatal on errors
 */
int
sudo_securid_verify(struct passwd *pw, const char *promp, sudo_auth *auth, struct sudo_conv_callback *callback)
{
    SDI_HANDLE *sd = (SDI_HANDLE *) auth->data;
    char *pass;
    int ret;
    debug_decl(sudo_securid_verify, SUDOERS_DEBUG_AUTH);

    pass = auth_getpass("Enter your PASSCODE: ", SUDO_CONV_PROMPT_ECHO_OFF,
	callback);

    /* Have ACE verify password */
    switch (SD_Check(*sd, pass, pw->pw_name)) {
	case ACM_OK:
		ret = AUTH_SUCESS;
		break;

	case ACE_UNDEFINED_PASSCODE:
		sudo_warnx("%s", U_("invalid passcode length for SecurID"));
		ret = AUTH_FATAL;
		break;

	case ACE_UNDEFINED_USERNAME:
		sudo_warnx("%s", U_("invalid username length for SecurID"));
		ret = AUTH_FATAL;
		break;

	case ACE_ERR_INVALID_HANDLE:
		sudo_warnx("%s", U_("invalid Authentication Handle for SecurID"));
		ret = AUTH_FATAL;
		break;

	case ACM_ACCESS_DENIED:
		ret = AUTH_FAILURE;
		break;

	case ACM_NEXT_CODE_REQUIRED:
                /* Sometimes (when current token close to expire?)
                   ACE challenges for the next token displayed
                   (entered without the PIN) */
		if (pass != NULL)
		    freezero(pass, strlen(pass));
        	pass = auth_getpass("\
!!! ATTENTION !!!\n\
Wait for the token code to change, \n\
then enter the new token code.\n", \
		SUDO_CONV_PROMPT_ECHO_OFF, callback);

		if (SD_Next(*sd, pass) == ACM_OK) {
			ret = AUTH_SUCCESS;
			break;
		}

		ret = AUTH_FAILURE;
		break;

	case ACM_NEW_PIN_REQUIRED:
                /*
		 * This user's SecurID has not been activated yet,
                 * or the pin has been reset
		 */
		/* XXX - Is setting up a new PIN within sudo's scope? */
		SD_Pin(*sd, "");
		sudo_printf(SUDO_CONV_ERROR_MSG|SUDO_CONV_PREFER_TTY, 
		    "Your SecurID access has not yet been set up.\n");
		sudo_printf(SUDO_CONV_ERROR_MSG|SUDO_CONV_PREFER_TTY, 
		    "Please set up a PIN before you try to authenticate.\n");
		ret = AUTH_FATAL;
		break;

	default:
		sudo_warnx("%s", U_("unknown SecurID error"));
		ret = AUTH_FATAL;
		break;
    }

    /* Free resources */
    SD_Close(*sd);

    if (pass != NULL)
	freezero(pass, strlen(pass));

    /* Return stored state to calling process */
    debug_return_int(ret);
}

#endif /* HAVE_SECURID */
