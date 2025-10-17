/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef _KERN_LOCK_ATTR_H_
#define _KERN_LOCK_ATTR_H_

#include <kern/lock_types.h>

__BEGIN_DECLS

#ifdef  XNU_KERNEL_PRIVATE
typedef struct _lck_attr_ {
	unsigned int    lck_attr_val;
} lck_attr_t;

extern lck_attr_t       lck_attr_default;

#define LCK_ATTR_NONE                   0
#define LCK_ATTR_DEBUG                  0x00000001
#define LCK_ATTR_RW_SHARED_PRIORITY     0x00010000
#else /* !XNU_KERNEL_PRIVATE */
typedef struct __lck_attr__ lck_attr_t;
#endif /* !XNU_KERNEL_PRIVATE */

#define LCK_ATTR_NULL ((lck_attr_t *)NULL)

extern  lck_attr_t      *lck_attr_alloc_init(void);

extern  void            lck_attr_setdefault(
	lck_attr_t              *attr);

extern  void            lck_attr_setdebug(
	lck_attr_t              *attr);

extern  void            lck_attr_cleardebug(
	lck_attr_t              *attr);

extern  void            lck_attr_free(
	lck_attr_t              *attr);

#ifdef  XNU_KERNEL_PRIVATE
/*!
 * @function lck_attr_rw_shared_priority
 *
 * @abstract
 * Changes the rw lock behaviour by setting reader priority over writer.
 *
 * @discussion
 * The attribute needs to be set before calling lck_rw_init().
 * This attribute changes the locking behaviour by possibly starving the writers.
 * Readers will always be able to lock the lock as long as a writer is not holding it.
 * This attribute was added to allow recursive locking in shared mode.
 *
 * @param attr	attr to modify
 */
extern  void            lck_attr_rw_shared_priority(
	lck_attr_t              *attr);
#endif /* XNU_KERNEL_PRIVATE */

__END_DECLS

#endif /* _KERN_LOCK_ATTR_H_ */
