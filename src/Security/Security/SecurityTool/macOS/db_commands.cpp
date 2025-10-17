/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#include "db_commands.h"

#include "readline_cssm.h"
#include "security_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <security_cdsa_client/dlclient.h>

using namespace CssmClient;

static int
do_db_create(const CSSM_GUID guid, const char *dbname, Boolean do_openparams, Boolean do_autocommit, Boolean do_mode, mode_t mode, Boolean do_version_0_params)
{
	int result = 0;

	try
	{
		CSSM_APPLEDL_OPEN_PARAMETERS openParameters = { sizeof(CSSM_APPLEDL_OPEN_PARAMETERS),
			(do_version_0_params ? 0u : CSSM_APPLEDL_OPEN_PARAMETERS_VERSION) };
		Cssm cssm;
		Module module(guid, cssm);
		DL dl(module);
		Db db(dl, dbname);

		if (do_openparams)
		{
			openParameters.autoCommit = do_autocommit;
			if (!do_version_0_params && do_mode)
			{
				openParameters.mask |= kCSSM_APPLEDL_MASK_MODE;
				openParameters.mode = mode;
			}

			db->openParameters(&openParameters);
		}

		db->create();
	}
	catch (const CommonError &e)
	{
		OSStatus status = e.osStatus();
		sec_error("CSSM_DbCreate %s: %s", dbname, sec_errstr(status));
	}
	catch (...)
	{
		result = 1;
	}

	return result;
}

static int
parse_guid(const char *name, CSSM_GUID *guid)
{
	size_t len = strlen(name);

	if (!strncmp("dl", name, len))
		*guid = gGuidAppleFileDL;
	else if (!strncmp("cspdl", name, len))
		*guid = gGuidAppleCSPDL;
	else
	{
		sec_error("Invalid guid: %s", name);
		return SHOW_USAGE_MESSAGE;
	}

	return 0;
}


static int
parse_mode(const char *name, mode_t *pmode)
{
	int result = 0;
	mode_t mode = 0;
	const char *p;

	if (!name || !pmode || *name != '0')
	{
		result = 2;
		goto loser;
	}

	for (p = name + 1; *p; ++p)
	{
		if (*p < '0' || *p > '7')
		{
			result = 2;
			goto loser;
		}

		mode = (mode << 3) + *p - '0';
	}

	*pmode = mode;
	return 0;

loser:
	sec_error("Invalid mode: %s", name);
	return result;
}

int
db_create(int argc, char * const *argv)
{
	char *dbname_to_free = NULL;
	char *dbname = NULL;
	int ch, result = 0;
	bool do_autocommit = true, do_mode = false;
	bool do_openparams = false, do_version_0_params = false;
	mode_t mode = 0666;
	CSSM_GUID guid = gGuidAppleFileDL;

	while ((ch = getopt(argc, argv, "0ahg:m:o")) != -1)
	{
		switch  (ch)
		{
		case '0':
			do_version_0_params = true;
			do_openparams = true;
			break;
		case 'a':
			do_autocommit = false;
			do_openparams = true;
			break;
		case 'g':
			result = parse_guid(optarg, &guid);
			if (result)
				goto loser;
			break;
		case 'm':
			result = parse_mode(optarg, &mode);
			if (result)
				goto loser;
			do_mode = true;
			do_openparams = true;
			break;
		case 'o':
			do_openparams = true;
			break;
		case '?':
		default:
			return SHOW_USAGE_MESSAGE;
		}
	}

	argc -= optind;
	argv += optind;

	if (argc > 0)
		dbname = *argv;
	else
	{
		fprintf(stderr, "db to create: ");
		dbname = readline(NULL, 0);
		if (!dbname)
		{
			result = -1;
			goto loser;
		}

		dbname_to_free = dbname;
		if (*dbname == '\0')
			goto loser;
	}

	do
	{
		result = do_db_create(guid, dbname, do_openparams, do_autocommit, do_mode, mode, do_version_0_params);
		if (result)
			goto loser;

		argc--;
		argv++;
		dbname = *argv;
	} while (argc > 0);

loser:
	if (dbname_to_free) {
		free(dbname_to_free);
	}

	return result;
}
