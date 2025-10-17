/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
 * FDSet.h
 * - maintains a list of file descriptors to watch and corresponding
 *   functions to call when the file descriptor is ready
 */

/* 
 * Modification History
 *
 * May 11, 2000		Dieter Siegmund (dieter@apple.com)
 * - created
 */


#ifndef _S_FDSET_H
#define _S_FDSET_H

#include <dispatch/dispatch.h>

/*
 * Type: FDCallout_func_t
 * Purpose:
 *   Client registers a function to call when file descriptor is ready.
 */

typedef void (FDCalloutFunc)(void * arg1, void * arg2);
typedef FDCalloutFunc * FDCalloutFuncRef;

struct FDCallout;
typedef struct FDCallout * FDCalloutRef;

FDCalloutRef
FDCalloutCreate(int fd, FDCalloutFuncRef func, void * arg1, void * arg2,
		dispatch_block_t cancel_handler);

void
FDCalloutRelease(FDCalloutRef * callout_p);

int
FDCalloutGetFD(FDCalloutRef callout);

void
FDCalloutSetDispatchQueue(dispatch_queue_t queue);

#endif /* _S_FDSET_H */
