/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
/* $Id: rwlock_test.c,v 1.26 2007/06/19 23:46:59 tbox Exp $ */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <isc/print.h>
#include <isc/thread.h>
#include <isc/rwlock.h>
#include <isc/string.h>
#include <isc/util.h>

#ifdef WIN32
#define sleep(x)	Sleep(1000 * x)
#endif

#ifdef ISC_PLATFORM_USETHREADS

isc_rwlock_t lock;

static isc_threadresult_t
#ifdef WIN32
WINAPI
#endif
run1(void *arg) {
	char *message = arg;

	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_read) ==
		      ISC_R_SUCCESS);
	printf("%s got READ lock\n", message);
	sleep(1);
	printf("%s giving up READ lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_read) ==
	       ISC_R_SUCCESS);
	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_read) ==
		      ISC_R_SUCCESS);
	printf("%s got READ lock\n", message);
	sleep(1);
	printf("%s giving up READ lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_read) ==
	       ISC_R_SUCCESS);
	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_write) ==
		      ISC_R_SUCCESS);
	printf("%s got WRITE lock\n", message);
	sleep(1);
	printf("%s giving up WRITE lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_write) ==
	       ISC_R_SUCCESS);
	return ((isc_threadresult_t)0);
}

static isc_threadresult_t
#ifdef WIN32
WINAPI
#endif
run2(void *arg) {
	char *message = arg;

	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_write) ==
		      ISC_R_SUCCESS);
	printf("%s got WRITE lock\n", message);
	sleep(1);
	printf("%s giving up WRITE lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_write) ==
	       ISC_R_SUCCESS);
	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_write) ==
		      ISC_R_SUCCESS);
	printf("%s got WRITE lock\n", message);
	sleep(1);
	printf("%s giving up WRITE lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_write) ==
	       ISC_R_SUCCESS);
	RUNTIME_CHECK(isc_rwlock_lock(&lock, isc_rwlocktype_read) ==
		      ISC_R_SUCCESS);
	printf("%s got READ lock\n", message);
	sleep(1);
	printf("%s giving up READ lock\n", message);
	RUNTIME_CHECK(isc_rwlock_unlock(&lock, isc_rwlocktype_read) ==
	       ISC_R_SUCCESS);
	return ((isc_threadresult_t)0);
}

int
main(int argc, char *argv[]) {
	unsigned int nworkers;
	unsigned int i;
	isc_thread_t workers[100];
	char name[100];
	void *dupname;

	if (argc > 1)
		nworkers = atoi(argv[1]);
	else
		nworkers = 2;
	if (nworkers > 100)
		nworkers = 100;
	printf("%d workers\n", nworkers);

	RUNTIME_CHECK(isc_rwlock_init(&lock, 5, 10) == ISC_R_SUCCESS);

	for (i = 0; i < nworkers; i++) {
		sprintf(name, "%02u", i);
		dupname = strdup(name);
		RUNTIME_CHECK(dupname != NULL);
		if (i != 0 && i % 3 == 0)
			RUNTIME_CHECK(isc_thread_create(run1, dupname,
							&workers[i]) ==
			       ISC_R_SUCCESS);
		else
			RUNTIME_CHECK(isc_thread_create(run2, dupname,
							&workers[i]) ==
			       ISC_R_SUCCESS);
	}

	for (i = 0; i < nworkers; i++)
		(void)isc_thread_join(workers[i], NULL);

	isc_rwlock_destroy(&lock);

	return (0);
}

#else

int
main(int argc, char *argv[]) {
	UNUSED(argc);
	UNUSED(argv);
	fprintf(stderr, "This test requires threads.\n");
	return(1);
}

#endif
