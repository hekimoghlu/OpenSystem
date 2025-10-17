/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#include <stdio.h>
#include <mach/mach.h>
#include <mach/message.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

static inline mach_port_type_t
get_port_type(mach_port_t mp)
{
	mach_port_type_t type = 0;
	mach_port_type(mach_task_self(), mp, &type);
	return type;
}

int
main()
{
	mach_port_t port = MACH_PORT_NULL;
	kern_return_t retval = KERN_SUCCESS;

	mach_port_t task = mach_task_self();

	printf("Starting the receive right allocation loop\n");
	int i = 0;
	while (!retval) {
		retval = mach_port_allocate(task, MACH_PORT_RIGHT_RECEIVE, &port);
		assert(retval == 0);
		//printf("%d. allocate a port=[%d]\n", i, port);
		assert(get_port_type(port) == MACH_PORT_TYPE_RECEIVE);
		i++;
	}

	exit(1);
}
