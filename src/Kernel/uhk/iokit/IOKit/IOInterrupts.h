/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 *  DRI: Josh de Cesare
 *
 */


#ifndef _IOKIT_IOINTERRUPTS_H
#define _IOKIT_IOINTERRUPTS_H

#define kIOInterruptTypeEdge  (0)
#define kIOInterruptTypeLevel (1)

#ifdef __cplusplus

class OSData;
class IOInterruptController;

struct IOInterruptSource {
	IOInterruptController *interruptController;
	OSData                *vectorData;
};
typedef struct IOInterruptSource IOInterruptSource;

#ifdef XNU_KERNEL_PRIVATE

struct IOInterruptSourcePrivate {
	void * vectorBlock;
};
typedef struct IOInterruptSourcePrivate IOInterruptSourcePrivate;

#endif /* XNU_KERNEL_PRIVATE */


#endif /* __cplusplus */

typedef void (*IOInterruptHandler)(void *target, void *refCon,
    void *nub, int source);

#endif /* ! _IOKIT_IOINTERRUPTS_H */
