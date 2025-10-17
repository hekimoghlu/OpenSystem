/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#import <objc/NSObject.h>

@protocol OS_dispatch_object <NSObject>
@end
@interface OS_dispatch_object: NSObject <OS_dispatch_object>
@end
typedef OS_dispatch_object *dispatch_object_t;

@protocol OS_dispatch_queue <OS_dispatch_object>
@end
@interface OS_dispatch_queue: OS_dispatch_object <OS_dispatch_object>
@end
typedef OS_dispatch_queue *dispatch_queue_t;

@protocol OS_dispatch_source <OS_dispatch_object>
@end
@interface OS_dispatch_source: OS_dispatch_object <OS_dispatch_object>
@end
typedef OS_dispatch_source *dispatch_source_t;

typedef void (^dispatch_block_t)(void);

dispatch_queue_t dispatch_get_current_queue(void);
void dispatch_async(dispatch_queue_t q, dispatch_block_t) __attribute__((nonnull));

void dispatch_sync(dispatch_queue_t q, 
                   __attribute__((noescape)) dispatch_block_t) __attribute__((nonnull));

void dispatch_retain(dispatch_object_t object) __attribute__((nonnull));
void dispatch_release(dispatch_object_t object) __attribute__((nonnull));
