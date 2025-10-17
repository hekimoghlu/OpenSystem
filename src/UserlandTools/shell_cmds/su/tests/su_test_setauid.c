/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#include <bsm/audit_session.h>

#include <err.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
	auditinfo_addr_t auinfo = {
		.ai_termid = { .at_type = AU_IPv4 },
		.ai_asid = AU_ASSIGN_ASID,
		.ai_auid = 1 /* daemon */,
		.ai_flags = 0,
	};

	if (getuid() != 0) {
		fprintf(stderr, "must be run as root\n");
		return (1);
	}

	if (argc < 2) {
		fprintf(stderr, "usage: %s [command ...]\n", getprogname());
		return (1);
	}

	/* Skip our argv[0], invoke argv[1]+ */
	argc -= 1;
	argv += 1;

	if (setaudit_addr(&auinfo, sizeof(auinfo)) != 0)
		err(1, "setaudit_addr");

	execvp(argv[0], argv);
	err(1, "execv");
}
