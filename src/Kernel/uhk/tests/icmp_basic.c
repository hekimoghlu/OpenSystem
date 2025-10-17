/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/errno.h>

#include <unistd.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"));

// NOTE: 17.253.144.10 is an anycast address run by AppleCDN.
// At the time of this test, this address is apple.com
static char *test_ip = "17.253.144.10";

static int
run_ping(char *dest, char *payload_size)
{
	int ping_ret, pid;
	// eg ping -t 3 -c 2 -s 120 17.253.144.10
	char *ping_args[]  = {"/sbin/ping", "-t", "3", "-c", "2", "-s", payload_size, dest, NULL};
	ping_ret = posix_spawn(&pid, ping_args[0], NULL, NULL, ping_args, NULL);
	if (ping_ret < 0) {
		return ping_ret;
	}
	waitpid(pid, &ping_ret, 0);
	return ping_ret;
}

static void
require_internet(void)
{
	int ret;

	ret = run_ping(test_ip, "128");
	if (ret == 0 || (WIFEXITED(ret) && !WEXITSTATUS(ret))) {
		T_PASS("Initial ping to %s passed, continuing test", test_ip);
	} else {
		T_SKIP("Initial ping to %s failed, skipping.", test_ip);
	}
}

T_DECL(icmp_internet_root, "test small Internet pings as root",
    T_META_ASROOT(true),
    T_META_REQUIRES_NETWORK(true))
{
	int ret;

	require_internet();

	ret = run_ping(test_ip, "10");
	if (ret == 0 || (WIFEXITED(ret) && !WEXITSTATUS(ret))) {
		T_PASS("ping completed");
	} else {
		T_FAIL("ping %s failed", test_ip);
	}
}

T_DECL(icmp_internet_non_root, "test small Internet pings as non-root",
    T_META_ASROOT(false),
    T_META_REQUIRES_NETWORK(true))
{
	int ret;

	require_internet();

	ret = run_ping(test_ip, "10");
	if (ret == 0 || (WIFEXITED(ret) && !WEXITSTATUS(ret))) {
		// And we did not crash
		T_PASS("ping completed");
	} else {
		T_FAIL("ping %s failed", test_ip);
	}
}

T_DECL(icmp_localhost_non_root, "test small localhost pings as non-root",
    T_META_ASROOT(false))
{
	int ret;
	ret = run_ping("127.0.0.1", "10");
	if (ret == 0 || (WIFEXITED(ret) && !WEXITSTATUS(ret))) {
		// And we did not crash
		T_PASS("ping completed");
	} else {
		T_FAIL("ping failed");
	}
}

T_DECL(icmp_localhost_root, "test small localhost pings as root",
    T_META_ASROOT(true))
{
	int ret;
	ret = run_ping("127.0.0.1", "10");
	if (ret == 0 || (WIFEXITED(ret) && !WEXITSTATUS(ret))) {
		T_PASS("ping completed");
	} else {
		T_FAIL("ping failed");
	}
}
