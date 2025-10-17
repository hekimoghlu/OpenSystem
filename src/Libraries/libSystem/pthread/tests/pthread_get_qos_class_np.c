/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>

#include <sys/qos.h>
#include <sys/resource.h>
#include <pthread.h>

#include "darwintest_defaults.h"

static void *sleep_thread(void __unused *arg){
	sleep(1);
	return NULL;
}

/* Regression test for <rdar://problem/29209770> */
T_DECL(test_pthread_get_qos_class_np, "Test for pthread_get_qos_class_np()", T_META_CHECK_LEAKS(NO)) {
	pthread_t thread;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_set_qos_class_np(&attr, QOS_CLASS_BACKGROUND, 0);
	pthread_create(&thread, &attr, sleep_thread, NULL);

	qos_class_t qos;
	pthread_get_qos_class_np(thread, &qos, NULL);

	T_EXPECT_EQ(qos, (qos_class_t)QOS_CLASS_BACKGROUND, NULL);
}
