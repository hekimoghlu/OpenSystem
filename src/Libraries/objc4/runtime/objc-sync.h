/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#ifndef __OBJC_SNYC_H_
#define __OBJC_SNYC_H_

#include <objc/objc.h>


/** 
 * Begin synchronizing on 'obj'.  
 * Allocates recursive pthread_mutex associated with 'obj' if needed.
 * 
 * @param obj The object to begin synchronizing on.
 * 
 * @return OBJC_SYNC_SUCCESS once lock is acquired.  
 */
OBJC_EXPORT int
objc_sync_enter(id _Nonnull obj)
    OBJC_AVAILABLE(10.3, 2.0, 9.0, 1.0, 2.0);

/** 
 * End synchronizing on 'obj'. 
 * 
 * @param obj The object to end synchronizing on.
 * 
 * @return OBJC_SYNC_SUCCESS or OBJC_SYNC_NOT_OWNING_THREAD_ERROR
 */
OBJC_EXPORT int
objc_sync_exit(id _Nonnull obj)
    OBJC_AVAILABLE(10.3, 2.0, 9.0, 1.0, 2.0);

enum {
    OBJC_SYNC_SUCCESS                 = 0,
    OBJC_SYNC_NOT_OWNING_THREAD_ERROR = -1
};


#endif // __OBJC_SYNC_H_
