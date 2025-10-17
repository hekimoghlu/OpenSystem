/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#ifdef HAVE_GETPRPWNAM

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#ifdef __hpux
#  undef MAXINT
#  include <hpsecurity.h>
#else
#  include <sys/security.h>
#endif /* __hpux */
#include <prot.h>

#include "sudoers.h"
#include "sudo_auth.h"

#ifdef __alpha
extern int crypt_type;
#endif

int
sudo_secureware_init(struct passwd *pw, sudo_auth *auth)
{
    debug_decl(sudo_secureware_init, SUDOERS_DEBUG_AUTH);

    /* Only initialize once. */
    if (auth->data != NULL)
	debug_return_int(AUTH_SUCCESS);

#ifdef __alpha
    if (crypt_type == INT_MAX)
	debug_return_int(AUTH_FAILURE);			/* no shadow */
#endif

    sudo_setspent();
    auth->data = sudo_getepw(pw);
    sudo_endspent();
    debug_return_int(auth->data ? AUTH_SUCCESS : AUTH_FATAL);
}

int
sudo_secureware_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback)
{
    char *pw_epasswd = auth->data;
    char *epass = NULL;
    debug_decl(sudo_secureware_verify, SUDOERS_DEBUG_AUTH);

    /* An empty plain-text password must match an empty encrypted password. */
    if (pass[0] == '\0')
	debug_return_int(pw_epasswd[0] ? AUTH_FAILURE : AUTH_SUCCESS);

#if defined(__alpha)
# ifdef HAVE_DISPCRYPT
	epass = dispcrypt(pass, pw_epasswd, crypt_type);
# else
	if (crypt_type == AUTH_CRYPT_BIGCRYPT)
	    epass = bigcrypt(pass, pw_epasswd);
	else if (crypt_type == AUTH_CRYPT_CRYPT16)
	    epass = crypt(pass, pw_epasswd);
# endif /* HAVE_DISPCRYPT */
#elif defined(HAVE_BIGCRYPT)
    epass = bigcrypt(pass, pw_epasswd);
#endif /* __alpha */

    if (epass != NULL && strcmp(pw_epasswd, epass) == 0)
	debug_return_int(AUTH_SUCCESS);
    debug_return_int(AUTH_FAILURE);
}

int
sudo_secureware_cleanup(struct passwd *pw, sudo_auth *auth, bool force)
{
    char *pw_epasswd = auth->data;
    debug_decl(sudo_secureware_cleanup, SUDOERS_DEBUG_AUTH);

    if (pw_epasswd != NULL)
	freezero(pw_epasswd, strlen(pw_epasswd));
    debug_return_int(AUTH_SUCCESS);
}

#endif /* HAVE_GETPRPWNAM */
