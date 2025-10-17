/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
/* $Id: AccountInfo.h,v 1.6 2007/06/19 23:47:07 tbox Exp $ */


#define RTN_OK		0
#define RTN_NOACCOUNT	1
#define RTN_NOMEMORY	2
#define RTN_ERROR	10

#define SE_SERVICE_LOGON_PRIV	L"SeServiceLogonRight"

/*
 * This routine retrieves the list of all Privileges associated with
 * a given account as well as the groups to which it beongs
 */
int
GetAccountPrivileges(
	char *name,			/* Name of Account */
	wchar_t **PrivList,		/* List of Privileges returned */
	unsigned int *PrivCount,	/* Count of Privileges returned */
	char **Groups,		/* List of Groups to which account belongs */
	unsigned int *totalGroups,	/* Count of Groups returned */
	int maxGroups		/* Maximum number of Groups to return */
	);

/*
 * This routine creates an account with the given name which has just
 * the logon service privilege and no membership of any groups,
 * i.e. it's part of the None group.
 */
BOOL
CreateServiceAccount(char *name, char *password);
