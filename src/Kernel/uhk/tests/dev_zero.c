/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include <fcntl.h>
#include <util.h>
#include <unistd.h>
#include <darwintest.h>

T_DECL(dev_zero,
    "test reading from /dev/zero",
    T_META_ASROOT(false))
{
	int dev = opendev("/dev/zero", O_RDONLY, 0, NULL);
	char buffer[100];

	for (int i = 0; i < 100; i++) {
		buffer[i] = 0xff;
	}

	int rd_sz = read(dev, buffer, sizeof(buffer));

	T_EXPECT_EQ(rd_sz, 100, "read from /dev/zero failed");

	for (int i = 0; i < 100; i++) {
		if (buffer[i]) {
			T_FAIL("Unexpected non-zero character read from /dev/zero");
		}
	}

	close(dev);
}
