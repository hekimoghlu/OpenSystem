/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
 * @OSF_COPYRIGHT@
 */
/*
 *	kern/host_statistics.h
 *
 *	Definitions for host VM/event statistics data structures.
 *
 */

#ifndef _KERN_HOST_STATISTICS_H_
#define _KERN_HOST_STATISTICS_H_

#include <kern/counter.h>

SCALABLE_COUNTER_DECLARE(vm_statistics_zero_fill_count);        /* # of zero fill pages */
SCALABLE_COUNTER_DECLARE(vm_statistics_reactivations);          /* # of pages reactivated */
SCALABLE_COUNTER_DECLARE(vm_statistics_pageins);                /* # of pageins */
SCALABLE_COUNTER_DECLARE(vm_statistics_pageouts);               /* # of pageouts */
SCALABLE_COUNTER_DECLARE(vm_statistics_faults);                 /* # of faults */
SCALABLE_COUNTER_DECLARE(vm_statistics_cow_faults);             /* # of copy-on-writes */
SCALABLE_COUNTER_DECLARE(vm_statistics_lookups);                /* object cache lookups */
SCALABLE_COUNTER_DECLARE(vm_statistics_hits);                   /* object cache hits */
SCALABLE_COUNTER_DECLARE(vm_statistics_purges);                 /* # of pages purged */
SCALABLE_COUNTER_DECLARE(vm_statistics_decompressions);         /* # of pages decompressed */
SCALABLE_COUNTER_DECLARE(vm_statistics_compressions);           /* # of pages compressed */
SCALABLE_COUNTER_DECLARE(vm_statistics_swapins);                /* # of pages swapped in (via compression segments) */
SCALABLE_COUNTER_DECLARE(vm_statistics_swapouts);               /* # of pages swapped out (via compression segments) */
SCALABLE_COUNTER_DECLARE(vm_statistics_total_uncompressed_pages_in_compressor); /* # of pages (uncompressed) held within the compressor. */

SCALABLE_COUNTER_DECLARE(vm_page_grab_count);

#endif  /* _KERN_HOST_STATISTICS_H_ */
