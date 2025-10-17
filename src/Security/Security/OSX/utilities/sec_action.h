/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#ifndef _SEC_ACTION_H_
#define _SEC_ACTION_H_

#include <dispatch/dispatch.h>

/*
 * Simple dispatch-based mechanism to coalesce high-frequency actions like
 * notifications. Sample usage:
 *
 *  static void
 *  notify_frequent_event(void)
 *  {
 *      static dispatch_once_t once;
 *      static sec_action_t action;
 *
 *      dispatch_once(&once, ^{
 *          action = sec_action_create("frequent_event", 2);
 *          sec_action_set_handler(action, ^{
 *              (void)notify_post("com.apple.frequent_event");
 *          });
 *      });
 *
 *      sec_action_perform(action);
 *  }
 *
 * The above will prevent com.apple.frequent_event from being posted more than
 * once every 2 seconds. For example, if notify_frequent_event is called 1000 times
 * over the span of 1.9s, the handler will be called twice, at 0s and 2s (approx).
 *
 * Default behavior is to perform actions on a queue with the same QOS as the caller.
 * If the action should be performed on a specific serial queue, the function
 * sec_action_create_with_queue can alternatively be used.
 */

typedef dispatch_source_t sec_action_t;

__BEGIN_DECLS

DISPATCH_MALLOC DISPATCH_RETURNS_RETAINED DISPATCH_WARN_RESULT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
sec_action_t
sec_action_create(const char *label, uint64_t interval);

DISPATCH_MALLOC DISPATCH_RETURNS_RETAINED DISPATCH_WARN_RESULT DISPATCH_NOTHROW
sec_action_t
sec_action_create_with_queue(dispatch_queue_t queue, const char *label, uint64_t interval);

DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
sec_action_set_handler(sec_action_t action, dispatch_block_t handler);

DISPATCH_NONNULL_ALL
void
sec_action_perform(sec_action_t action);

__END_DECLS

#endif /* _SEC_ACTION_H_ */
