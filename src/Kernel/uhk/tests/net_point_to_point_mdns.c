/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("dlil"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_ASROOT(true));

T_DECL(point_to_point_mdn_utun,
    "Point-to-point mDNS on utun interface", T_META_TAG_VM_PREFERRED)
{
	int udp_fd;
	int tun_fd;
	struct ctl_info kernctl_info = { 0 };
	struct sockaddr_ctl kernctl_addr = { 0 };
	char ifname[IFXNAMSIZ];
	socklen_t optlen;
	struct ifreq ifr = { 0 };

	T_QUIET; T_EXPECT_POSIX_SUCCESS(udp_fd = socket(AF_INET, SOCK_DGRAM, 0), NULL);

	T_ASSERT_POSIX_SUCCESS(tun_fd = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL), NULL);

	strlcpy(kernctl_info.ctl_name, UTUN_CONTROL_NAME, sizeof(kernctl_info.ctl_name));
	T_ASSERT_POSIX_SUCCESS(ioctl(tun_fd, CTLIOCGINFO, &kernctl_info), NULL);

	kernctl_addr.sc_len = sizeof(kernctl_addr);
	kernctl_addr.sc_family = AF_SYSTEM;
	kernctl_addr.ss_sysaddr = AF_SYS_CONTROL;
	kernctl_addr.sc_id = kernctl_info.ctl_id;
	kernctl_addr.sc_unit = 0;

	T_ASSERT_POSIX_SUCCESS(bind(tun_fd, (struct sockaddr *)&kernctl_addr, sizeof(kernctl_addr)), NULL);

	T_ASSERT_POSIX_SUCCESS(connect(tun_fd, (struct sockaddr *)&kernctl_addr, sizeof(kernctl_addr)), NULL);

	optlen = sizeof(ifname);
	T_ASSERT_POSIX_SUCCESS(getsockopt(tun_fd, SYSPROTO_CONTROL, UTUN_OPT_IFNAME, ifname, &optlen), NULL);
	T_LOG("utun interface: %s", ifname);

	strlcpy(ifr.ifr_name, ifname, sizeof(ifr.ifr_name));
	T_ASSERT_POSIX_SUCCESS(ioctl(udp_fd, SIOCGPOINTOPOINTMDNS, &ifr), NULL);
	T_ASSERT_EQ_INT(ifr.ifr_point_to_point_mdns, 0, "point_to_point_mdns off by default");

	ifr.ifr_point_to_point_mdns = 1;
	T_ASSERT_POSIX_SUCCESS(ioctl(udp_fd, SIOCSPOINTOPOINTMDNS, &ifr), NULL);
	T_ASSERT_POSIX_SUCCESS(ioctl(udp_fd, SIOCGPOINTOPOINTMDNS, &ifr), NULL);
	T_ASSERT_EQ_INT(ifr.ifr_point_to_point_mdns, 1, "point_to_point_mdns turned on");

	ifr.ifr_point_to_point_mdns = 0;
	T_ASSERT_POSIX_SUCCESS(ioctl(udp_fd, SIOCSPOINTOPOINTMDNS, &ifr), NULL);
	T_ASSERT_POSIX_SUCCESS(ioctl(udp_fd, SIOCGPOINTOPOINTMDNS, &ifr), NULL);
	T_ASSERT_EQ_INT(ifr.ifr_point_to_point_mdns, 0, "point_to_point_mdns turned off");

	T_ASSERT_POSIX_SUCCESS(close(tun_fd), NULL);
	T_ASSERT_POSIX_SUCCESS(close(udp_fd), NULL);
}
