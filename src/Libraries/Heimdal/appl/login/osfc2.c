/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#include "login_locl.h"
RCSID("$Id$");

int
do_osfc2_magic(uid_t uid)
{
#ifdef HAVE_OSFC2
    struct es_passwd *epw;
    char *argv[2];

    /* fake */
    argv[0] = (char*)getprogname();
    argv[1] = NULL;
    set_auth_parameters(1, argv);

    epw = getespwuid(uid);
    if(epw == NULL) {
	syslog(LOG_AUTHPRIV|LOG_NOTICE,
	       "getespwuid failed for %d", uid);
	printf("Sorry.\n");
	return 1;
    }
    /* We don't check for auto-retired, foo-retired,
       bar-retired, or any other kind of retired accounts
       here; neither do we check for time-locked accounts, or
       any other kind of serious C2 mumbo-jumbo. We do,
       however, call setluid, since failing to do so is not
       very good (take my word for it). */

    if(!epw->uflg->fg_uid) {
	syslog(LOG_AUTHPRIV|LOG_NOTICE,
	       "attempted login by %s (has no uid)", epw->ufld->fd_name);
	printf("Sorry.\n");
	return 1;
    }
    setluid(epw->ufld->fd_uid);
    if(getluid() != epw->ufld->fd_uid) {
	syslog(LOG_AUTHPRIV|LOG_NOTICE,
	       "failed to set LUID for %s (%d)",
	       epw->ufld->fd_name, epw->ufld->fd_uid);
	printf("Sorry.\n");
	return 1;
    }
#endif /* HAVE_OSFC2 */
    return 0;
}
