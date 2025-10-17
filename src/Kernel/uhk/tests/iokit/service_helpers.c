/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#include <errno.h>
#include <stdio.h>

#include "service_helpers.h"

#define MAX_RETRIES 10

/*
 * Helper method to find IOServices needed for testing. Use with T_ASSERT_POSIX_SUCCESS(...)
 */
int
IOTestServiceFindService(const char * name, io_service_t * serviceOut)
{
	int err = 0;
	int retries = 0;
	io_service_t service = IO_OBJECT_NULL;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
	err = sysctlbyname("kern.iokit_test_service_setup", NULL, 0, (void *)name, strlen(name));
#pragma clang diagnostic pop
	if (err) {
		goto finish;
	}

	do {
		service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching(name));
		if (service == IO_OBJECT_NULL) {
			sleep(1);
			retries += 1;
		}
	} while (service == IO_OBJECT_NULL && retries <= MAX_RETRIES);

	if (service == IO_OBJECT_NULL) {
		err = ENOENT;
		goto finish;
	}

	err = 0;

finish:
	if (serviceOut && service != IO_OBJECT_NULL) {
		*serviceOut = service;
	} else if (service != IO_OBJECT_NULL) {
		IOObjectRelease(service);
	}

	return err;
}
