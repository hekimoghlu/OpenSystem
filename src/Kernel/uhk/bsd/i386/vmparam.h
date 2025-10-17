/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#ifndef _BSD_I386_VMPARAM_H_
#define _BSD_I386_VMPARAM_H_ 1

#if defined (__i386__) || defined (__x86_64__)

#include <sys/resource.h>

#define USRSTACK        VM_USRSTACK32
#define USRSTACK64      VM_USRSTACK64


/*
 * Virtual memory related constants, all in bytes
 */
#ifndef DFLDSIZ
#define DFLDSIZ         (RLIM_INFINITY)         /* initial data size limit */
#endif
#ifndef MAXDSIZ
#define MAXDSIZ         (RLIM_INFINITY)         /* max data size */
#endif
#ifndef DFLSSIZ
#define DFLSSIZ         (8*1024*1024)           /* initial stack size limit */
#endif
#ifndef MAXSSIZ
#define MAXSSIZ         (64*1024*1024)          /* max stack size */
#endif
#ifndef DFLCSIZ
#define DFLCSIZ         (0)                     /* initial core size limit */
#endif
#ifndef MAXCSIZ
#define MAXCSIZ         (RLIM_INFINITY)         /* max core size */
#endif  /* MAXCSIZ */

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif  /* _BSD_I386_VMPARAM_H_ */
