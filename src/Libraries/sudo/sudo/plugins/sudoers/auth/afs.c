/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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

#ifdef HAVE_AFS

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

#include <afs/stds.h>
#include <afs/kautils.h>

#include "sudoers.h"
#include "sudo_auth.h"
#include "check.h"

int
sudo_afs_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback)
{
    struct ktc_encryptionKey afs_key;
    struct ktc_token afs_token;
    debug_decl(sudo_afs_verify, SUDOERS_DEBUG_AUTH);

    if (IS_NONINTERACTIVE(auth))
	debug_return_int(AUTH_NONINTERACTIVE);

    /* Display lecture if needed and we haven't already done so. */
    display_lecture(callback);

    /* Try to just check the password */
    ka_StringToKey(pass, NULL, &afs_key);
    if (ka_GetAdminToken(pw->pw_name,		/* name */
			 NULL,			/* instance */
			 NULL,			/* realm */
			 &afs_key,		/* key (contains password) */
			 0,			/* lifetime */
			 &afs_token,		/* token */
			 0) == 0)		/* new */
	debug_return_int(AUTH_SUCCESS);

    /* Fall back on old method XXX - needed? */
    setpag();
    if (ka_UserAuthenticateGeneral(KA_USERAUTH_VERSION+KA_USERAUTH_DOSETPAG,
				   pw->pw_name,	/* name */
				   NULL,	/* instance */
				   NULL,	/* realm */
				   pass,	/* password */
				   0,		/* lifetime */
				   NULL,	/* expiration ptr (unused) */
				   0,		/* spare */
				   NULL) == 0)	/* reason */
	debug_return_int(AUTH_SUCCESS);

    debug_return_int(AUTH_FAILURE);
}

#endif HAVE_AFS
