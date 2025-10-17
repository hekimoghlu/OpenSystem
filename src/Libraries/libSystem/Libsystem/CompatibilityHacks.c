/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
/*
 * See: <rdar://problem/8289209>, <rdar://problem/8351271>, <rdar://problem/8359348>
 */

#include <TargetConditionals.h>

#if defined(__x86_64__) && !TARGET_OS_DRIVERKIT

#define SYM(sym) \
  __asm__(".globl R8289209$_" #sym "; R8289209$_" #sym ": jmp _" #sym)

/****************/

SYM(close);
SYM(fork);
SYM(fsync);
SYM(getattrlist);
SYM(getrlimit);
SYM(getxattr);
SYM(open);
SYM(pthread_attr_destroy);
SYM(pthread_attr_init);
SYM(pthread_attr_setdetachstate);
SYM(pthread_create);
SYM(pthread_mutex_lock);
SYM(pthread_mutex_unlock);
SYM(pthread_self);
SYM(ptrace);
SYM(read);
SYM(setattrlist);
SYM(setrlimit);
SYM(sigaction);
SYM(stat);
SYM(sysctl);
SYM(time);
SYM(unlink);
SYM(write);

#endif /* defined(__x86_64__) && !TARGET_OS_DRIVERKIT*/

