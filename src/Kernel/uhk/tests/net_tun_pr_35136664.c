/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/kern_control.h>
#include <sys/sys_domain.h>

#include <net/if_utun.h>
#include <net/if_ipsec.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(T_META_NAMESPACE("xnu.net"),
    T_META_RUN_CONCURRENTLY(true),
    T_META_ASROOT(true));

T_DECL(PR_35136664_utun,
    "This bind a utun and close it without connecting", T_META_TAG_VM_PREFERRED)
{
	int tunsock;
	struct ctl_info kernctl_info;
	struct sockaddr_ctl kernctl_addr;

	T_ASSERT_POSIX_SUCCESS(tunsock = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL), NULL);

	memset(&kernctl_info, 0, sizeof(kernctl_info));
	strlcpy(kernctl_info.ctl_name, UTUN_CONTROL_NAME, sizeof(kernctl_info.ctl_name));
	T_ASSERT_POSIX_SUCCESS(ioctl(tunsock, CTLIOCGINFO, &kernctl_info), NULL);

	memset(&kernctl_addr, 0, sizeof(kernctl_addr));
	kernctl_addr.sc_len = sizeof(kernctl_addr);
	kernctl_addr.sc_family = AF_SYSTEM;
	kernctl_addr.ss_sysaddr = AF_SYS_CONTROL;
	kernctl_addr.sc_id = kernctl_info.ctl_id;
	kernctl_addr.sc_unit = 0;

	T_ASSERT_POSIX_SUCCESS(bind(tunsock, (struct sockaddr *)&kernctl_addr, sizeof(kernctl_addr)), NULL);

	T_ASSERT_POSIX_SUCCESS(close(tunsock), NULL);
}

T_DECL(PR_35136664_ipsec,
    "This bind a ipsec and close it without connecting", T_META_TAG_VM_PREFERRED)
{
	int tunsock;
	struct ctl_info kernctl_info;
	struct sockaddr_ctl kernctl_addr;

	T_ASSERT_POSIX_SUCCESS(tunsock = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL), NULL);

	memset(&kernctl_info, 0, sizeof(kernctl_info));
	strlcpy(kernctl_info.ctl_name, IPSEC_CONTROL_NAME, sizeof(kernctl_info.ctl_name));
	T_ASSERT_POSIX_SUCCESS(ioctl(tunsock, CTLIOCGINFO, &kernctl_info), NULL);

	memset(&kernctl_addr, 0, sizeof(kernctl_addr));
	kernctl_addr.sc_len = sizeof(kernctl_addr);
	kernctl_addr.sc_family = AF_SYSTEM;
	kernctl_addr.ss_sysaddr = AF_SYS_CONTROL;
	kernctl_addr.sc_id = kernctl_info.ctl_id;
	kernctl_addr.sc_unit = 0;

	T_ASSERT_POSIX_SUCCESS(bind(tunsock, (struct sockaddr *)&kernctl_addr, sizeof(kernctl_addr)), NULL);

	T_ASSERT_POSIX_SUCCESS(close(tunsock), NULL);
}
