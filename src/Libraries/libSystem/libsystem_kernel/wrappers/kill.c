/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#include <signal.h>

extern int __kill(pid_t pid, int sig, int posix);

/*
 * kill stub, which wraps a modified kill system call that takes a posix
 * behaviour indicator as the third parameter to indicate whether or not
 * conformance to standards is needed.  We use a trailing parameter in
 * case the call is called directly via syscall(), since for most uses,
 * it won't matter to the caller.
 */
int
kill(pid_t pid, int sig)
{
#if __DARWIN_UNIX03
	return __kill(pid, sig, 1);
#else   /* !__DARWIN_UNIX03 */
	return __kill(pid, sig, 0);
#endif  /* !__DARWIN_UNIX03 */
}
