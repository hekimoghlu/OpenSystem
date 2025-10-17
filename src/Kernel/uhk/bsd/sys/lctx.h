/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifndef _SYS_LCTX_H_
#define _SYS_LCTX_H_

#ifndef KERNEL
#include <sys/errno.h> /* errno, ENOSYS */
#include <sys/_types/_pid_t.h> /* pid_t */
static __inline pid_t
getlcid(pid_t pid)
{
	errno = ENOSYS;
	return -1;
}

static __inline int
setlcid(pid_t pid, pid_t lcid)
{
	errno = ENOSYS;
	return -1;
}
#endif

#define LCID_PROC_SELF  (0)
#define LCID_REMOVE     (-1)
#define LCID_CREATE     (0)

#endif  /* !_SYS_LCTX_H_ */
