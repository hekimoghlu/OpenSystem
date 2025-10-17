/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#include <pthread.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <libkern/OSAtomic.h>

pthread_t thr;
pthread_rwlock_t lock;
OSSpinLock slock = 0;
int i = 0;

void sighandler(int sig)
{
	if (sig == SIGUSR1) {
		OSSpinLockLock(&slock);
		OSSpinLockUnlock(&slock);		
	} else {
		// ALARM
		fprintf(stderr, "FAIL (%d)\n", i);
		exit(1);
	}
}

void* thread(void *arg)
{
	pthread_rwlock_rdlock(&lock);
	pthread_rwlock_unlock(&lock);
	return NULL;
}

int main(int argc, const char *argv[])
{
	pthread_rwlock_init(&lock, NULL);
	signal(SIGUSR1, sighandler);
	signal(SIGALRM, sighandler);

	alarm(30);

	while (i++ < 10000) {
		pthread_rwlock_wrlock(&lock);
		pthread_create(&thr, NULL, thread, NULL);
		OSSpinLockLock(&slock);
		pthread_kill(thr, SIGUSR1);
		pthread_rwlock_unlock(&lock);
		OSSpinLockUnlock(&slock);
	}
	
	fprintf(stderr, "PASS\n");
	return 0;
}
