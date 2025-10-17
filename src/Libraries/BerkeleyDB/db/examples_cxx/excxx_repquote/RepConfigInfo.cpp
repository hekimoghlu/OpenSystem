/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#include "RepConfigInfo.h"

RepConfigInfo::RepConfigInfo()
{
	start_policy = DB_REP_ELECTION;
	home = "TESTDIR";
	got_listen_address = false;
	totalsites = 0;
	priority = 100;
	verbose = false;
	other_hosts = NULL;
}

RepConfigInfo::~RepConfigInfo()
{
	// release any other_hosts structs.
	if (other_hosts != NULL) {
		REP_HOST_INFO *CurItem = other_hosts;
		while (CurItem->next != NULL)
		{
			REP_HOST_INFO *TmpItem = CurItem;
			free(CurItem);
			CurItem = TmpItem;
		}
		free(CurItem);
	}
	other_hosts = NULL;
}

void RepConfigInfo::addOtherHost(char* host, int port, bool peer)
{
	REP_HOST_INFO *newinfo;
	newinfo = (REP_HOST_INFO*)malloc(sizeof(REP_HOST_INFO));
	newinfo->host = host;
	newinfo->port = port;
	newinfo->peer = peer;
	if (other_hosts == NULL) {
		other_hosts = newinfo;
		newinfo->next = NULL;
	} else {
		newinfo->next = other_hosts;
		other_hosts = newinfo;
	}
}
