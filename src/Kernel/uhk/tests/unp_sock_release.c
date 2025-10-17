/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#include <nfs/nfs.h>

#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_ASROOT(true)
	);

T_DECL(test_unp_sock_release, "UDS with sock_release()")
{
	int fds[2] = { -1, -1 };
	struct nfsd_args nfsd_args = { 0 };

	T_ASSERT_POSIX_SUCCESS(socketpair(AF_UNIX, SOCK_DGRAM, 0, fds),
	    "socketpair(AF_UNIX, SOCK_DGRAM, 0)");
	T_LOG("socketpair() fds: %d, %d\n", fds[0], fds[1]);

	nfsd_args.sock = fds[0];
	T_ASSERT_POSIX_SUCCESS(nfssvc(NFSSVC_EXPORT | NFSSVC_ADDSOCK | NFSSVC_NFSD, &nfsd_args),
	    "nfssvc() sock %d", fds[0]);

	T_ASSERT_POSIX_SUCCESS(close(fds[0]), "close(%d)\n", fds[0]);

	T_ASSERT_POSIX_SUCCESS(nfssvc(NFSSVC_EXPORT | NFSSVC_NFSD, NULL), "nfssvc() NULL");

	T_ASSERT_POSIX_SUCCESS(close(fds[1]), "close(%d)\n", fds[1]);
}
