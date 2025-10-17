/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#ifndef __firehose_h
#define __firehose_h

__BEGIN_DECLS

/*!
 * @function __firehose_buffer_push_to_logd
 *
 * @abstract
 * Called by the dispatch firehose apis to notify logd that a chunk is available
 */
__WATCHOS_AVAILABLE(3.0) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0)
void __firehose_buffer_push_to_logd(firehose_buffer_t fb, bool for_io);

/*!
 * @function __firehose_allocate
 *
 * @abstract
 * Wrapper to allocate kernel memory
 */
__WATCHOS_AVAILABLE(3.0) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0)
void __firehose_allocate(vm_offset_t *addr, vm_size_t size);

/*!
 * @function __firehose_critical_region_enter
 *
 * @abstract
 * Function that disables preemption
 */
__WATCHOS_AVAILABLE(3.0) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0)
extern void __firehose_critical_region_enter(void);

/*!
 * @function __firehose_critical_region_leave
 *
 * @abstract
 * Function that enables preemption
 */
__WATCHOS_AVAILABLE(3.0) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0)
extern void __firehose_critical_region_leave(void);

extern void oslogwakeup(void);

__END_DECLS

#endif /* __firehose_h */
