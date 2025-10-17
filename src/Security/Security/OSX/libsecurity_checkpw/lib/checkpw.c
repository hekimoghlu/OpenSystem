/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include <security/pam_appl.h>
#include <security/openpam.h>

#include "checkpw.h"
#include <syslog.h>
#include <unistd.h>

#define PAM_STACK_NAME "checkpw"

static
int checkpw_internal_pam( const char* uname, const char* password )
{
	int checkpwret = CHECKPW_FAILURE;

	int pamret = PAM_SUCCESS;
	pam_handle_t *pamh;
	struct pam_conv pamc;
	pamc.conv = &openpam_nullconv;

	pamret = pam_start(PAM_STACK_NAME, uname, &pamc, &pamh);
	if (PAM_SUCCESS != pamret)
	{
		syslog(LOG_WARNING,"PAM: Unable to start pam.");
		goto pamerr_no_end;
	}

	pamret = pam_set_item(pamh, PAM_AUTHTOK, password);
	if (PAM_SUCCESS != pamret)
	{
		syslog(LOG_WARNING,"PAM: Unable to set password.");
		goto pamerr;
	}

	pamret = pam_authenticate(pamh, 0);
	if (PAM_SUCCESS != pamret)
	{
		syslog(LOG_WARNING,"PAM: Unable to authenticate.");
		checkpwret = CHECKPW_BADPASSWORD;
		goto pamerr;
	}

	pamret = pam_acct_mgmt(pamh, 0);
	if (PAM_SUCCESS != pamret)
	{
		if (PAM_NEW_AUTHTOK_REQD == pamret)
		{
			syslog(LOG_WARNING,"PAM: Unable to authorize, password needs to be changed.");
		} else {
			syslog(LOG_WARNING,"PAM: Unable to authorize.");
		}

		goto pamerr;
	}

	checkpwret = CHECKPW_SUCCESS;

pamerr:
	pam_end(pamh, pamret);
pamerr_no_end:
	return checkpwret;

}

int checkpw_internal( const struct passwd* pw, const char* password );
int checkpw_internal( const struct passwd* pw, const char* password )
{
	return checkpw(pw->pw_name, password);
}

int checkpw( const char* userName, const char* password )
{
	int				siResult = CHECKPW_FAILURE;
	// workaround for 3965234; I assume the empty string is OK...
	const char	   *thePassword = password ? password : "";

	if (userName == NULL)
		return CHECKPW_UNKNOWNUSER;
	
	siResult = checkpw_internal_pam(userName, thePassword);
	switch (siResult) {
		case CHECKPW_SUCCESS:
		case CHECKPW_UNKNOWNUSER:
		case CHECKPW_BADPASSWORD:
			break;
		default:
			usleep(500000);
			siResult = checkpw_internal_pam(userName, thePassword);
			break;
	}
	
	return siResult;
}

