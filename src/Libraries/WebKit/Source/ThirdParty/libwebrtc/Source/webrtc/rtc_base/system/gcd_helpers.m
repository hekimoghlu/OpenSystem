/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "rtc_base/system/gcd_helpers.h"

dispatch_queue_t RTCDispatchQueueCreateWithTarget(const char *label,
                                                  dispatch_queue_attr_t attr,
                                                  dispatch_queue_t target) {
  if (@available(iOS 10, macOS 10.12, tvOS 10, watchOS 3, *)) {
    return dispatch_queue_create_with_target(label, attr, target);
  }
  dispatch_queue_t queue = dispatch_queue_create(label, attr);
  dispatch_set_target_queue(queue, target);
  return queue;
}
