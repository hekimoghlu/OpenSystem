/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#ifndef __NETINET_IN_STAT_H__
#define __NETINET_IN_STAT_H__

#ifdef PRIVATE

#include <stdint.h>

typedef struct activity_bitmap {
	uint64_t        start;          /* Start timestamp using uptime */
	uint64_t        bitmap[2];      /* 128 bit map, each bit == 8 sec */
} activity_bitmap_t;

#endif /* PRIVATE */

#ifdef BSD_KERNEL_PRIVATE

extern void in_stat_set_activity_bitmap(activity_bitmap_t *activity, uint64_t now);

#endif /* BSD_KERNEL_PRIVATE */

#endif /* __NETINET_IN_STAT_H__ */
