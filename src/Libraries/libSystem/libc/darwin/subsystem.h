/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
#ifndef __SUBSYSTEM_H__
#define __SUBSYSTEM_H__

#include <sys/stat.h>

__BEGIN_DECLS
/*
 * Returns an fd for the given path, relative to root or to
 * the subsystem root for the process.  Behaves exactly like
 * open in every way, except O_CREAT is forbidden.
 *
 * Returns a file descriptor on success, or -1 on failure.
 * errno is set exactly as open would have set it, except
 * that O_CREAT will result in EINVAL.
 */
int open_with_subsystem(const char * path, int oflag);

/*
 * Invokes stat for the given path, relative to root or to
 * the subsystem root for the process.  Behaves exactly like
 * stat in every way.
 *
 * Returns 0 on success, or -1 on failure.  On failure, errno
 * is set exactly as stat would have set it.
 */
int stat_with_subsystem(const char *__restrict path, struct stat *__restrict buf);
__END_DECLS

#endif /* __SUBSYSTEM_H__ */
