/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#include <bsm/audit.h>

/*
 * Wrappers for the legacy getaudit() and setaudit() syscalls.
 */

int
getaudit(struct auditinfo *ainfo)
{
	int err;
	auditinfo_addr_t aia;

	if ((err = getaudit_addr(&aia, sizeof(aia))) != 0) {
		return err;
	}

	ainfo->ai_auid = aia.ai_auid;
	ainfo->ai_mask = aia.ai_mask;
	ainfo->ai_termid.port = aia.ai_termid.at_port;
	ainfo->ai_termid.machine = aia.ai_termid.at_addr[0];
	ainfo->ai_asid = aia.ai_asid;

	return 0;
}

int
setaudit(const struct auditinfo *ainfo)
{
	int err;
	struct auditinfo *ai = (struct auditinfo *)ainfo;
	auditinfo_addr_t aia;

	/* Get the current ai_flags so they are preserved. */
	if ((err = getaudit_addr(&aia, sizeof(aia))) != 0) {
		return err;
	}

	aia.ai_auid = ai->ai_auid;
	aia.ai_mask = ai->ai_mask;
	aia.ai_termid.at_port = ai->ai_termid.port;
	aia.ai_termid.at_type = AU_IPv4;
	aia.ai_termid.at_addr[0] = ai->ai_termid.machine;
	aia.ai_asid = ai->ai_asid;

	if ((err = setaudit_addr(&aia, sizeof(aia))) != 0) {
		return err;
	}

	/* The session ID may have been assigned by kernel so copy that back. */
	ai->ai_asid = aia.ai_asid;

	return 0;
}
