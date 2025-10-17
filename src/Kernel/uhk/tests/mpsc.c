/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#ifdef T_NAMESPACE
#undef T_NAMESPACE
#endif

#include <darwintest.h>
#include <sys/sysctl.h>

T_GLOBAL_META(T_META_NAMESPACE("xnu.mpsc"),
    T_META_RUN_CONCURRENTLY(true));

T_DECL(pingpong, "mpsc_pingpong", T_META_ASROOT(true))
{
	uint64_t count = 100 * 1000, nsecs = 0;
	size_t nlen = sizeof(nsecs);
	int error;

	error = sysctlbyname("kern.mpsc_test_pingpong", &nsecs, &nlen,
	    &count, sizeof(count));
	T_ASSERT_POSIX_SUCCESS(error, "sysctlbyname");
	T_LOG("%lld asyncs in %lld ns (%g us/async)", count, nsecs,
	    (nsecs / 1e3) / count);
}
