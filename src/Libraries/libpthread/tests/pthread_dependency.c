/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#include "darwintest_defaults.h"
#include <darwintest_utils.h>
#include <pthread/dependency_private.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static struct job {
	pthread_dependency_t *req;
	useconds_t usleep;
	int done;
} job;

static void *
do_test(void *__unused arg)
{
	pthread_mutex_lock(&mutex);

	while (!job.done) {
		while (job.req == 0) {
			pthread_cond_wait(&cond, &mutex);
		}
		if (job.usleep) usleep(job.usleep);
		pthread_dependency_fulfill_np(job.req, job.req);
		job.req = NULL;
	}

	pthread_mutex_unlock(&mutex);
	return NULL;
}

static void
post_req(pthread_dependency_t *req, useconds_t delay, bool done)
{
	pthread_mutex_lock(&mutex);
	job.req = req;
	job.usleep = delay;
	job.done = done;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
}

T_DECL(dependency, "dependency", T_META_ALL_VALID_ARCHS(YES))
{
	pthread_dependency_t req;
	pthread_t pth;
	void *v;
	int ret;

	T_ASSERT_POSIX_ZERO(pthread_create(&pth, NULL, do_test, NULL), NULL);

	T_LOG("Waiting on a pdependency that takes some time");

	pthread_dependency_init_np(&req, pth, NULL);
	post_req(&req, 100000, false);
	v = pthread_dependency_wait_np(&req);
	T_EXPECT_EQ(v, &req, "pthread_dependency_wait worked");

	T_LOG("Waiting on a pdependency that is already fulfilled");

	pthread_dependency_init_np(&req, pth, NULL);
	post_req(&req, 0, false);
	usleep(100000);
	v = pthread_dependency_wait_np(&req);
	T_EXPECT_EQ(v, &req, "pthread_dependency_wait worked");

	T_LOG("Waiting on a fulfilled pdependency with the other thread exiting");

	pthread_dependency_init_np(&req, pth, NULL);
	post_req(&req, 0, true);
	ret = pthread_join(pth, NULL);
	T_EXPECT_POSIX_ZERO(ret, "pthread_join");

	v = pthread_dependency_wait_np(&req);
	T_EXPECT_EQ(v, &req, "pthread_dependency_wait worked");

	T_END;
}
