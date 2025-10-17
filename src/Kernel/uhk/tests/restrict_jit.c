/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/mman.h>

#include <darwintest.h>


/*
 * macOS only test. Try to map 2 different MAP_JIT regions. 2nd should fail.
 */
T_DECL(restrict_jit, "macOS restricted JIT entitlement test")
{
#if TARGET_OS_OSX
	void *addr1;
	void *addr2;
	size_t size = 64 * 1024;


	addr1 = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0);
	T_ASSERT_NE_PTR(addr1, MAP_FAILED, "First map MAP_JIT");

	addr2 = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0);
	if (addr2 == MAP_FAILED) {
		T_PASS("Only one MAP_JIT was allowed");
	} else {
		T_FAIL("Second MAP_JIT was allowed");
	}

#else
	T_SKIP("Not macOS");
#endif
}
