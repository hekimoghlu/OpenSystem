/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
/* $Id: named-journalprint.c,v 1.2 2009/12/04 21:59:23 marka Exp $ */

/*! \file */
#include <config.h>

#include <isc/log.h>
#include <isc/mem.h>
#include <isc/print.h>
#include <isc/util.h>

#include <dns/journal.h>
#include <dns/log.h>
#include <dns/result.h>
#include <dns/types.h>

#include <stdlib.h>

/*
 * Setup logging to use stderr.
 */
static isc_result_t
setup_logging(isc_mem_t *mctx, FILE *errout, isc_log_t **logp) {
	isc_logdestination_t destination;
	isc_logconfig_t *logconfig = NULL;
	isc_log_t *log = NULL;

	RUNTIME_CHECK(isc_log_create(mctx, &log, &logconfig) == ISC_R_SUCCESS);
	isc_log_setcontext(log);
	dns_log_init(log);
	dns_log_setcontext(log);

	destination.file.stream = errout;
	destination.file.name = NULL;
	destination.file.versions = ISC_LOG_ROLLNEVER;
	destination.file.maximum_size = 0;
	RUNTIME_CHECK(isc_log_createchannel(logconfig, "stderr",
					    ISC_LOG_TOFILEDESC,
					    ISC_LOG_DYNAMIC,
					    &destination, 0) == ISC_R_SUCCESS);
	RUNTIME_CHECK(isc_log_usechannel(logconfig, "stderr",
					 NULL, NULL) == ISC_R_SUCCESS);

	*logp = log;
	return (ISC_R_SUCCESS);
}

int
main(int argc, char **argv) {
	char *file;
	isc_mem_t *mctx = NULL;
	isc_result_t result;
	isc_log_t *lctx = NULL;

	if (argc != 2) {
		printf("usage: %s journal\n", argv[0]);
		return(1);
	}

	file = argv[1];

	RUNTIME_CHECK(isc_mem_create(0, 0, &mctx) == ISC_R_SUCCESS);
	RUNTIME_CHECK(setup_logging(mctx, stderr, &lctx) == ISC_R_SUCCESS);

	result = dns_journal_print(mctx, file, stdout);
	if (result == DNS_R_NOJOURNAL)
		fprintf(stderr, "%s\n", dns_result_totext(result));
	isc_log_destroy(&lctx);
	isc_mem_detach(&mctx);
	return(result != ISC_R_SUCCESS ? 1 : 0);
}
