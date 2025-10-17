/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
b_txn(int argc, char *argv[])
{
	extern char *optarg;
	extern int optind;
	DB_ENV *dbenv;
	DB_TXN *txn;
	int tabort, ch, i, count;

	count = 1000;
	tabort = 0;
	while ((ch = getopt(argc, argv, "ac:")) != EOF)
		switch (ch) {
		case 'a':
			tabort = 1;
			break;
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

	/* Create the environment. */
	DB_BENCH_ASSERT(db_env_create(&dbenv, 0) == 0);
	dbenv->set_errfile(dbenv, stderr);
#if DB_VERSION_MAJOR == 3 && DB_VERSION_MINOR < 1
	DB_BENCH_ASSERT(dbenv->open(dbenv, TESTDIR,
	    NULL, DB_CREATE | DB_INIT_LOCK | DB_INIT_LOG |
	    DB_INIT_MPOOL | DB_INIT_TXN | DB_PRIVATE, 0666) == 0);
#else
	DB_BENCH_ASSERT(dbenv->open(dbenv, TESTDIR,
	    DB_CREATE | DB_INIT_LOCK | DB_INIT_LOG |
	    DB_INIT_MPOOL | DB_INIT_TXN | DB_PRIVATE, 0666) == 0);
#endif

	/* Start and commit/abort a transaction count times. */
	TIMER_START;
	if (tabort)
		for (i = 0; i < count; ++i) {
#if DB_VERSION_MAJOR < 4
			DB_BENCH_ASSERT(txn_begin(dbenv, NULL, &txn, 0) == 0);
			DB_BENCH_ASSERT(txn_abort(txn) == 0);
#else
			DB_BENCH_ASSERT(
			    dbenv->txn_begin(dbenv, NULL, &txn, 0) == 0);
			DB_BENCH_ASSERT(txn->abort(txn) == 0);
#endif
		}
	else
		for (i = 0; i < count; ++i) {
#if DB_VERSION_MAJOR < 4
			DB_BENCH_ASSERT(txn_begin(dbenv, NULL, &txn, 0) == 0);
			DB_BENCH_ASSERT(txn_commit(txn, 0) == 0);
#else
			DB_BENCH_ASSERT(
			    dbenv->txn_begin(dbenv, NULL, &txn, 0) == 0);
			DB_BENCH_ASSERT(txn->commit(txn, 0) == 0);
#endif
		}
	TIMER_STOP;

	printf("# %d empty transaction start/%s pairs\n",
	    count, tabort ? "abort" : "commit");
	TIMER_DISPLAY(count);

	DB_BENCH_ASSERT(dbenv->close(dbenv, 0) == 0);

	return (0);
}

static int
usage()
{
	(void)fprintf(stderr, "usage: b_txn [-a] [-c count]\n");
	return (EXIT_FAILURE);
}

