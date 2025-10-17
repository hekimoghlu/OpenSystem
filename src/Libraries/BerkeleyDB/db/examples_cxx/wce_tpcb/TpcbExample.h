/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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


#ifndef _TPCBEXAMPLE_H_INCLUDE__
#define	_TPCBEXAMPLE_H_INCLUDE__

#include <windows.h>
#include "db.h"

#define	ACCOUNTS	    1000
#define	BRANCHES	      10
#define	TELLERS		     100
#define	HISTORY		   10000
#define	TRANSACTIONS	1000
#define	TESTDIR		"TESTDIR"

typedef enum { ACCOUNT, BRANCH, TELLER } FTYPE;

extern "C" {
void tpcb_errcallback(const DB_ENV *, const char *, const char *);
}

class TpcbExample
{
public:
	int createEnv(int);
	void closeEnv();
	int populate();
	int run(int);
	int txn(DB *, DB *, DB *, DB *,	int, int, int);
	int populateHistory(DB *, int, u_int32_t, u_int32_t, u_int32_t);
	int populateTable(DB *, u_int32_t, u_int32_t, int, const char *);

	TpcbExample();

	char *getHomeDir(char *, int);
	wchar_t *getHomeDirW(wchar_t *, int);
	void setHomeDir(char *);
	void setHomeDirW(wchar_t *);

#define	ERR_STRING_MAX 1024
	char msgString[ERR_STRING_MAX];
	int accounts;
	int branches;
	int history;
	int tellers;

	// options configured through the advanced dialog.
	int fast_mode;
	int verbose;
	int cachesize;
	int rand_seed;
private:
	DB_ENV *dbenv;
	char homeDirName[MAX_PATH];
	wchar_t wHomeDirName[MAX_PATH];

	u_int32_t randomId(FTYPE, u_int32_t, u_int32_t, u_int32_t);
	u_int32_t randomInt(u_int32_t, u_int32_t);
	// no need for copy and assignment
	TpcbExample(const TpcbExample &);
	void operator = (const TpcbExample &);
};

#endif
