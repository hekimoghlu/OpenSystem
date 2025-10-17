/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
 * Copyright (c) 1997, Apple Computer, Inc. All rights reserved.
 *
 */

#ifndef _BSD_I386_PROFILE_H_
#define _BSD_I386_PROFILE_H_

#if defined (__i386__) || defined (__x86_64__)

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_UNSTABLE

/*
 * Block interrupts during mcount so that those interrupts can also be
 * counted (as soon as we get done with the current counting).  On the
 * i386 platfom, can't do splhigh/splx as those are C routines and can
 * recursively invoke mcount.
 */
#warning MCOUNT_* not implemented yet.

#define MCOUNT_INIT
#define MCOUNT_ENTER    /* s = splhigh(); */ /* XXX TODO */
#define MCOUNT_EXIT     /* (void) splx(s); */ /* XXX TODO */

#endif /* __APPLE_API_UNSTABLE */
#endif /* KERNEL */

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif /* _BSD_I386_PROFILE_H_ */
