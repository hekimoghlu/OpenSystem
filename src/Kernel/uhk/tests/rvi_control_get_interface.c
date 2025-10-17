/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/sys_domain.h>
#include <sys/kern_control.h>

#include <darwintest.h>

#define RVI_CONTROL_NAME          "com.apple.net.rvi_control"
#define RVI_COMMAND_GET_INTERFACE 0x20

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_ENABLED(TARGET_OS_OSX),
	T_META_ASROOT_(1)
	);

T_DECL(rvi_control_get_interface, "getsockopt on RVI control-socket triggering out-of-bounds memory access", T_META_TAG_VM_PREFERRED)
{
	int fd;

	T_ASSERT_POSIX_SUCCESS(fd = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL), NULL);

	struct ctl_info ctl_info = {
		.ctl_name = RVI_CONTROL_NAME
	};
	T_ASSERT_POSIX_SUCCESS(ioctl(fd, CTLIOCGINFO, &ctl_info), NULL);

	struct sockaddr_ctl sockaddr_ctl = {
		.sc_len = sizeof(struct sockaddr_ctl),
		.sc_family = AF_SYSTEM,
		.ss_sysaddr = AF_SYS_CONTROL,
		.sc_id = ctl_info.ctl_id,
		.sc_unit = 0
	};
	T_ASSERT_POSIX_SUCCESS(connect(fd, (const struct sockaddr *)&sockaddr_ctl, sizeof(struct sockaddr_ctl)), NULL);

	char data[10];
	socklen_t data_len = 1;
	T_ASSERT_POSIX_SUCCESS(getsockopt(fd, SYSPROTO_CONTROL, RVI_COMMAND_GET_INTERFACE, &data, &data_len), NULL);

	data_len = 5;
	T_ASSERT_POSIX_SUCCESS(getsockopt(fd, SYSPROTO_CONTROL, RVI_COMMAND_GET_INTERFACE, &data, &data_len), NULL);
	T_ASSERT_EQ(data_len, 5, "data_len == 5", NULL);

	data_len = 10;
	T_ASSERT_POSIX_SUCCESS(getsockopt(fd, SYSPROTO_CONTROL, RVI_COMMAND_GET_INTERFACE, &data, &data_len), NULL);
	T_ASSERT_EQ(data_len, 5, "data_len == 5", NULL);

	T_PASS("success");
}
