/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include <stdio.h>

#include <security/pam_appl.h>
#include <security/openpam.h>	/* for openpam_ttyconv() */

#include "passwd.h"

extern char* progname;
static pam_handle_t *pamh;
static struct pam_conv pamc;

//-------------------------------------------------------------------------------------
//	pam_check_passwd
//-------------------------------------------------------------------------------------

int
pam_check_passwd(char* uname)
{
	int retval = PAM_SUCCESS;

	/* Initialize PAM. */
	pamc.conv = &openpam_ttyconv;
	pam_start(progname, uname, &pamc, &pamh);

	printf("Checking password for %s.\n", uname);

	/* Authenticate. */
	if (PAM_SUCCESS != (retval = pam_authenticate(pamh, 0)))
		goto pamerr;

	/* Authorize. */
	if (PAM_SUCCESS != (retval = pam_acct_mgmt(pamh, 0)) && PAM_NEW_AUTHTOK_REQD != retval)
		goto pamerr;

	/* Change the password. */
	if (PAM_NEW_AUTHTOK_REQD == retval && PAM_SUCCESS != (retval = pam_chauthtok(pamh, 0)))
		goto pamerr;

	/* Set the credentials. */
	if (PAM_SUCCESS != (retval = pam_setcred(pamh, PAM_ESTABLISH_CRED)))
		goto pamerr;

	/* Open the session. */
	if (PAM_SUCCESS != (retval = pam_open_session(pamh, 0)))
		goto pamerr;

	/* Close the session. */
	if (PAM_SUCCESS != (retval = pam_close_session(pamh, 0)))
		goto pamerr;

pamerr:
	/* Print an error, if needed. */
	if (PAM_SUCCESS != retval)
		fprintf(stderr, "Sorry\n");

	/* Terminate PAM. */
	pam_end(pamh, retval);
	return retval;
}
