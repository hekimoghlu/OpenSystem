/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#ifdef HAVE_GETSPNAM
# include <shadow.h>
#endif /* HAVE_GETSPNAM */
#ifdef HAVE_GETPRPWNAM
# ifdef __hpux
#  undef MAXINT
#  include <hpsecurity.h>
# else
#  include <sys/security.h>
# endif /* __hpux */
# include <prot.h>
#endif /* HAVE_GETPRPWNAM */

#include "sudoers.h"

/*
 * Exported for auth/secureware.c
 */
#if defined(HAVE_GETPRPWNAM) && defined(__alpha)
int crypt_type = INT_MAX;
#endif /* HAVE_GETPRPWNAM && __alpha */

/*
 * Return a copy of the encrypted password for the user described by pw.
 * If shadow passwords are in use, look in the shadow file.
 */
char *
sudo_getepw(const struct passwd *pw)
{
    char *epw = NULL;
    debug_decl(sudo_getepw, SUDOERS_DEBUG_AUTH);

    /* If there is a function to check for shadow enabled, use it... */
#ifdef HAVE_ISCOMSEC
    if (!iscomsec())
	goto done;
#endif /* HAVE_ISCOMSEC */

#ifdef HAVE_GETPWNAM_SHADOW
    {
	struct passwd *spw;

	/* On OpenBSD we need to closed the non-shadow passwd db first. */
	endpwent();
	if ((spw = getpwnam_shadow(pw->pw_name)) != NULL)
	    epw = spw->pw_passwd;
	setpassent(1);
    }
#endif /* HAVE_GETPWNAM_SHADOW */
#ifdef HAVE_GETPRPWNAM
    {
	struct pr_passwd *spw;

	if ((spw = getprpwnam(pw->pw_name)) && spw->ufld.fd_encrypt) {
# ifdef __alpha
	    crypt_type = spw->ufld.fd_oldcrypt;
# endif /* __alpha */
	    epw = spw->ufld.fd_encrypt;
	}
    }
#endif /* HAVE_GETPRPWNAM */
#ifdef HAVE_GETSPNAM
    {
	struct spwd *spw;

	if ((spw = getspnam(pw->pw_name)) && spw->sp_pwdp)
	    epw = spw->sp_pwdp;
    }
#endif /* HAVE_GETSPNAM */

#if defined(HAVE_ISCOMSEC)
done:
#endif
    /* If no shadow password, fall back on regular password. */
    debug_return_str(strdup(epw ? epw : pw->pw_passwd));
}

void
sudo_setspent(void)
{
    debug_decl(sudo_setspent, SUDOERS_DEBUG_AUTH);

#ifdef HAVE_GETPRPWNAM
    setprpwent();
#endif
#ifdef HAVE_GETSPNAM
    setspent();
#endif
    debug_return;
}

void
sudo_endspent(void)
{
    debug_decl(sudo_endspent, SUDOERS_DEBUG_AUTH);

#ifdef HAVE_GETPRPWNAM
    endprpwent();
#endif
#ifdef HAVE_GETSPNAM
    endspent();
#endif
    debug_return;
}
