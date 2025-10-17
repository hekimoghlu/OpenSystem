/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#include <sys/ucred.h>
#include <sys/un.h>

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <darwintest.h>

static char buffer[LINE_MAX];

#define FILE_PATH "/tmp/uds-to-self.sock"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false));


T_DECL(uds_self_connection, "self-connecting Unix domain sockets")
{
	int fd;
	struct sockaddr_un sun = { 0 };
	socklen_t solen;
	ssize_t nsent;
	ssize_t nrcvd;
	struct xucred xucred;
	pid_t pid;
	uuid_t uuid;
	audit_token_t token;

	T_ASSERT_POSIX_SUCCESS(fd = socket(AF_UNIX, SOCK_DGRAM, 0), NULL);

	sun.sun_family = AF_UNIX;
	snprintf(sun.sun_path, sizeof(sun.sun_path), FILE_PATH);
	sun.sun_len = (unsigned char) SUN_LEN(&sun);

	unlink(FILE_PATH);

	T_ASSERT_POSIX_SUCCESS(bind(fd, (struct sockaddr *)&sun, sun.sun_len), NULL);

	solen = sizeof(struct sockaddr_un);
	T_ASSERT_POSIX_SUCCESS(getsockname(fd, (struct sockaddr *)&sun, &solen), NULL);
	T_LOG("socket bound to %s", sun.sun_path);

	T_ASSERT_POSIX_SUCCESS(connect(fd, (struct sockaddr *)&sun, sun.sun_len), NULL);

	T_ASSERT_POSIX_SUCCESS(getpeername(fd, (struct sockaddr *)&sun, &solen), NULL);
	T_LOG("socket connected to %s", sun.sun_path);

	T_ASSERT_POSIX_SUCCESS(nsent = send(fd, buffer, strlen(buffer) + 1, 0), NULL);
	T_LOG("send %ld bytes\n", nsent);

	T_ASSERT_POSIX_SUCCESS(nrcvd = recv(fd, buffer, sizeof(buffer), 0), NULL);

	/*
	 * The return value of getsockopt() is not important, what matters is to not panic
	 */
	solen = sizeof(xucred);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEERCRED, &xucred, &solen);

	solen = sizeof(pid);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEERPID, &pid, &solen);

	solen = sizeof(pid);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEEREPID, &pid, &solen);

	solen = sizeof(uuid);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEERUUID, &uuid, &solen);

	solen = sizeof(uuid);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEEREUUID, &uuid, &solen);

	solen = sizeof(token);
	(void)getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &solen);

	T_ASSERT_POSIX_SUCCESS(shutdown(fd, SHUT_RDWR), NULL);

	T_ASSERT_POSIX_SUCCESS(close(fd), NULL);

	unlink(FILE_PATH);
}
