/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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

#if defined(HAVE_SKEY) || defined(HAVE_OPIE)

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

#if defined(HAVE_SKEY)
# include <skey.h>
# define RFC1938				skey
#  ifdef HAVE_RFC1938_SKEYCHALLENGE
#   define rfc1938challenge(a,b,c,d)	skeychallenge((a),(b),(c),(d))
#  else
#   define rfc1938challenge(a,b,c,d)	skeychallenge((a),(b),(c))
#  endif
# define rfc1938verify(a,b)		skeyverify((a),(b))
#elif defined(HAVE_OPIE)
# include <opie.h>
# define RFC1938			opie
# define rfc1938challenge(a,b,c,d)	opiechallenge((a),(b),(c))
# define rfc1938verify(a,b)		opieverify((a),(b))
#endif

#include "sudoers.h"
#include "sudo_auth.h"

int
sudo_rfc1938_setup(struct passwd *pw, char **promptp, sudo_auth *auth)
{
    char challenge[256];
    size_t challenge_len;
    static char *orig_prompt = NULL, *new_prompt = NULL;
    static size_t op_len, np_size;
    static struct RFC1938 rfc1938;
    debug_decl(sudo_rfc1938_setup, SUDOERS_DEBUG_AUTH);

    /* Stash a pointer to the rfc1938 struct if we have not initialized */
    if (!auth->data)
	auth->data = &rfc1938;

    /* Save the original prompt */
    if (orig_prompt == NULL) {
	orig_prompt = *promptp;
	op_len = strlen(orig_prompt);

	/* Ignore trailing colon (we will add our own) */
	if (orig_prompt[op_len - 1] == ':')
	    op_len--;
	else if (op_len >= 2 && orig_prompt[op_len - 1] == ' '
	    && orig_prompt[op_len - 2] == ':')
	    op_len -= 2;
    }

#ifdef HAVE_SKEY
    /* Close old stream */
    if (rfc1938.keyfile)
	(void) fclose(rfc1938.keyfile);
#endif

    /*
     * Look up the user and get the rfc1938 challenge.
     * If the user is not in the OTP db, only post a fatal error if
     * we are running alone (since they may just use a normal passwd).
     */
    if (rfc1938challenge(&rfc1938, pw->pw_name, challenge, sizeof(challenge))) {
	if (IS_ONEANDONLY(auth)) {
	    sudo_warnx(U_("you do not exist in the %s database"), auth->name);
	    debug_return_int(AUTH_FATAL);
	} else {
	    debug_return_int(AUTH_FAILURE);
	}
    }

    /* Get space for new prompt with embedded challenge */
    challenge_len = strlen(challenge);
    if (np_size < op_len + challenge_len + 7) {
	char *p = realloc(new_prompt, op_len + challenge_len + 7);
	if (p == NULL) {
	    sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	    debug_return_int(AUTH_FATAL);
	}
	np_size = op_len + challenge_len + 7;
	new_prompt = p;
    }

    if (def_long_otp_prompt)
	(void) snprintf(new_prompt, np_size, "%s\n%s", challenge, orig_prompt);
    else
	(void) snprintf(new_prompt, np_size, "%.*s [ %s ]:", (int)op_len,
	    orig_prompt, challenge);

    *promptp = new_prompt;
    debug_return_int(AUTH_SUCCESS);
}

int
sudo_rfc1938_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback)
{
    debug_decl(sudo_rfc1938_verify, SUDOERS_DEBUG_AUTH);

    if (rfc1938verify((struct RFC1938 *) auth->data, (char *)pass) == 0)
	debug_return_int(AUTH_SUCCESS);
    else
	debug_return_int(AUTH_FAILURE);
}

#endif /* HAVE_SKEY || HAVE_OPIE */
