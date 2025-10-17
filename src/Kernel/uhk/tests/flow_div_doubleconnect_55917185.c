/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include <stdlib.h>

#include <pthread.h>
#include <sys/kern_control.h>
#include <sys/socket.h>
#include <sys/sys_domain.h>
#include <unistd.h>

#include <darwintest.h>

/* we should win the race in this window: */
#define NTRIES 200000

static void *
connect_race(void *data)
{
	int *ps = data;
	struct sockaddr_ctl sc = {
		.sc_id = 1 /* com.apple.flow-divert */
	};
	int n;

	for (n = 0; n < NTRIES; ++n) {
		connect(*ps, (const struct sockaddr *)&sc, sizeof(sc));
	}

	return NULL;
}

T_DECL(flow_div_doubleconnect_55917185, "Bad error path in double-connect for flow_divert_kctl_connect", T_META_TAG_VM_PREFERRED)
{
	int s = -1;
	int tmp_s;
	struct sockaddr_ctl sc = {
		.sc_id = 1 /* com.apple.flow-divert */
	};
	pthread_t t;
	int n;

	T_SETUPBEGIN;
	T_ASSERT_POSIX_ZERO(pthread_create(&t, NULL, connect_race, &s), NULL);
	T_SETUPEND;

	for (n = 0; n < NTRIES; ++n) {
		T_ASSERT_POSIX_SUCCESS(tmp_s = socket(AF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL), NULL);

		/*
		 * this bind will fail, but that's ok because it initialises
		 * kctl:
		 */
		bind(tmp_s, (const struct sockaddr *)&sc, sizeof(sc));

		/* this is what we're racing the other thread for: */
		s = tmp_s;
		connect(s, (const struct sockaddr *)&sc, sizeof(sc));

		T_ASSERT_POSIX_SUCCESS(close(s), NULL);
		s = -1;
	}

	T_ASSERT_POSIX_ZERO(pthread_join(t, NULL), NULL);
	T_PASS("flow_divert_kctl_connect race didn't trigger panic");
}
