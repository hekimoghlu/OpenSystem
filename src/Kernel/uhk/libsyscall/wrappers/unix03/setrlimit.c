/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#include <sys/cdefs.h>

#if __DARWIN_UNIX03

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

extern int __setrlimit(int resource, const struct rlimit *rlp);

/*
 * setrlimit stub, for conformance, OR in _RLIMIT_POSIX_FLAG
 *
 * This is for UNIX03 only.
 */
int
setrlimit(int resource, const struct rlimit *rlp)
{
	resource |= _RLIMIT_POSIX_FLAG;
	return __setrlimit(resource, rlp);
}

#endif /* __DARWIN_UNIX03 */
