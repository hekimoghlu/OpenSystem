/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#include <string.h>
#include <netinet/in_stat.h>

#define IN_STAT_ACTIVITY_GRANULARITY            8       /* 8 sec granularity */
#define IN_STAT_ACTIVITY_TIME_SEC_SHIFT         3       /* 8 sec per bit */
#define IN_STAT_ACTIVITY_BITMAP_TOTAL_SIZE      ((uint64_t) 128)
#define IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE      ((uint64_t) 64)
#define IN_STAT_ACTIVITY_TOTAL_TIME             ((uint64_t) (8 * 128))
#define IN_STAT_SET_MOST_SIGNIFICANT_BIT        ((u_int64_t )0x8000000000000000)

void
in_stat_set_activity_bitmap(activity_bitmap_t *activity, uint64_t now)
{
	uint64_t elapsed_time, slot;
	uint64_t *bitmap;
	if (activity->start == 0) {
		// Align all activity maps
		activity->start = now - (now % IN_STAT_ACTIVITY_GRANULARITY);
	}
	elapsed_time = now - activity->start;

	slot = elapsed_time >> IN_STAT_ACTIVITY_TIME_SEC_SHIFT;
	if (slot < IN_STAT_ACTIVITY_BITMAP_TOTAL_SIZE) {
		if (slot < IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE) {
			bitmap = &activity->bitmap[0];
		} else {
			bitmap = &activity->bitmap[1];
			slot -= IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE;
		}
		*bitmap |= (((u_int64_t) 1) << slot);
	} else {
		if (slot >= (IN_STAT_ACTIVITY_BITMAP_TOTAL_SIZE * 2)) {
			activity->start = now - IN_STAT_ACTIVITY_TOTAL_TIME;
			activity->bitmap[0] = activity->bitmap[1] = 0;
		} else {
			uint64_t shift =
			    slot - (IN_STAT_ACTIVITY_BITMAP_TOTAL_SIZE - 1);
			/*
			 * Move the start time and bitmap forward to
			 * cover the lost time
			 */
			activity->start +=
			    (shift << IN_STAT_ACTIVITY_TIME_SEC_SHIFT);
			if (shift > IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE) {
				activity->bitmap[0] = activity->bitmap[1];
				activity->bitmap[1] = 0;
				shift -= IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE;
				if (shift == IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE) {
					activity->bitmap[0] = 0;
				} else {
					activity->bitmap[0] >>= shift;
				}
			} else {
				uint64_t mask_lower, tmp;
				uint64_t b1_low, b0_high;

				/*
				 * generate a mask with all of lower
				 * 'shift' bits set
				 */
				tmp = (((uint64_t)1) << (shift - 1));
				mask_lower = ((tmp - 1) ^ tmp);
				activity->bitmap[0] >>= shift;
				b1_low = (activity->bitmap[1] & mask_lower);

				b0_high = (b1_low <<
				    (IN_STAT_ACTIVITY_BITMAP_FIELD_SIZE -
				    shift));
				activity->bitmap[0] |= b0_high;
				activity->bitmap[1] >>= shift;
			}
		}
		activity->bitmap[1] |= IN_STAT_SET_MOST_SIGNIFICANT_BIT;
	}
}
