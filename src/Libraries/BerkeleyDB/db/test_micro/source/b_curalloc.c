/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "bench.h"

static int usage(void);

int
b_curalloc(int argc, char *argv[])
{
	extern char *optarg;
	extern int optind;
	DB *dbp;
	DBC *curp;
	int ch, i, count;

	count = 100000;
	while ((ch = getopt(argc, argv, "c:")) != EOF)
		switch (ch) {
		case 'c':
			count = atoi(optarg);
			break;
		case '?':
		default:
			return (usage());
		}
	argc -= optind;
	argv += optind;
	if (argc != 0)
		return (usage());

	/* Create the database. */
	DB_BENCH_ASSERT(db_create(&dbp, NULL, 0) == 0);
	dbp->set_errfile(dbp, stderr);

#if DB_VERSION_MAJOR >= 4 && DB_VERSION_MINOR >= 1
	DB_BENCH_ASSERT(dbp->open(
	    dbp, NULL, TESTFILE, NULL, DB_BTREE, DB_CREATE, 0666) == 0);
#else
	DB_BENCH_ASSERT(
	    dbp->open(dbp, TESTFILE, NULL, DB_BTREE, DB_CREATE, 0666) == 0);
#endif

	/* Allocate a cursor count times. */
	TIMER_START;
	for (i = 0; i < count; ++i) {
		DB_BENCH_ASSERT(dbp->cursor(dbp, NULL, &curp, 0) == 0);
		DB_BENCH_ASSERT(curp->c_close(curp) == 0);
	}
	TIMER_STOP;

	printf("# %d cursor allocations\n", count);
	TIMER_DISPLAY(count);

	DB_BENCH_ASSERT(dbp->close(dbp, 0) == 0);

	return (0);
}

static int
usage()
{
	(void)fprintf(stderr, "usage: b_curalloc [-c count]\n");
	return (EXIT_FAILURE);
}
