/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include <sys/param.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <err.h>
#include <stdio.h>
#include <unistd.h>
#include <strings.h>
#include <stdlib.h>
#include <sysexits.h>

#include <stdint.h>
#include <netsmb/smb.h>

#include <smbclient/smbclient.h>
#include <smbclient/smbclient_internal.h>
#include <smbclient/smbclient_netfs.h>
#include <smbclient/ntstatus.h>

#include "SetNetworkAccountSID.h"
#include "LsarLookup.h"

#define MAX_SID_PRINTBUFFER	256	/* Used to print out the sid in case of an error */
static
void print_ntsid(ntsid_t *sidptr, const char *account, const char *domain)
{
	char sidprintbuf[MAX_SID_PRINTBUFFER];
	char *s = sidprintbuf;
	int subs;
	uint64_t auth = 0;
	unsigned i;
	uint32_t *ip;
	size_t len;
	
	bzero(sidprintbuf, MAX_SID_PRINTBUFFER);
	for (i = 0; i < sizeof(sidptr->sid_authority); i++)
		auth = (auth << 8) | sidptr->sid_authority[i];
	s += snprintf(s, MAX_SID_PRINTBUFFER, "S-%u-%llu", sidptr->sid_kind, auth);
	
	subs = sidptr->sid_authcount;
	
	for (ip = sidptr->sid_authorities; subs--; ip++)  { 
		len = MAX_SID_PRINTBUFFER - (s - sidprintbuf);
		s += snprintf(s, len, "-%u", *ip); 
	}
	os_log_debug(OS_LOG_DEFAULT, "%s\\%s network sid %s \n",
			   (domain) ? domain : "", (account) ? account : "", sidprintbuf);
}

void setNetworkAccountSID(void *sessionRef, void *args) 
{
#pragma unused(args)
	SMBHANDLE serverConnection = SMBAllocateAndSetContext(sessionRef);
	ntsid_t *ntsid = NULL;
	SMBServerPropertiesV1 properties;
	NTSTATUS status;
	char *account = NULL, *domain = NULL;
	
	if (!serverConnection) {
		goto done;
	}
	status = SMBGetServerProperties(serverConnection, &properties, kPropertiesVersion, sizeof(properties));
	if (!NT_SUCCESS(status)) {
		goto done;
	}
	/* We already have a network sid assigned, then do nothing */
	if (properties.internalFlags & kHasNtwrkSID) {
		goto done;
	}
	
	/* We never set the user sid if guest or anonymous authentication */
	if ((properties.authType == kSMBAuthTypeGuest) || (properties.authType == kSMBAuthTypeAnonymous)) {
		goto done;
	}
	status = GetNetworkAccountSID(properties.serverName, &account, &domain, &ntsid);
	if (!NT_SUCCESS(status)) {
		goto done;
	}
	print_ntsid(ntsid, account, domain);
	/* 
	 * In the future this should return an ntstatus and set errno. Currently we
	 * ignore the error, since the failure just means ACLs are off. 
	 */
	(void)SMBSetNetworkIdentity(serverConnection, ntsid, account, domain);
done:
	if (account) {
		free(account);
	}
	if (domain) {
		free(domain);
	}
	if (ntsid) {
		free(ntsid);
	}
	if (serverConnection) {
		free(serverConnection);
	}
}
