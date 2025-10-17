/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
 * @APPLE_FREE_COPYRIGHT@
 */

#ifndef _ARM_RTCLOCK_H_
#define _ARM_RTCLOCK_H_

#include <mach/boolean.h>
#include <mach/mach_types.h>
#include <mach/mach_time.h>
#include <arm/machine_routines.h>

#define EndOfAllTime            0xFFFFFFFFFFFFFFFFULL
#define DECREMENTER_MAX         0x7FFFFFFFUL
#define DECREMENTER_MIN         0xAUL

typedef struct _rtclock_data_ {
	uint32_t                                                rtc_sec_divisor;
	uint32_t                                                rtc_usec_divisor;
	mach_timebase_info_data_t               rtc_timebase_const;
	union {
		uint64_t                abstime;
		struct {
			uint32_t        low;
			uint32_t        high;
		} abstime_val;
	}                                                               rtc_base;
	union {
		uint64_t                abstime;
		struct {
			uint32_t        low;
			uint32_t        high;
		} abstime_val;
	}                                                               rtc_adj;
	tbd_ops_data_t                                  rtc_timebase_func;

	/* Only needed for AIC manipulation */
	vm_offset_t                                             rtc_timebase_addr;
	vm_offset_t                                             rtc_timebase_val;
} rtclock_data_t;

extern rtclock_data_t                                   RTClockData;
#define rtclock_sec_divisor                             RTClockData.rtc_sec_divisor
#define rtclock_usec_divisor                    RTClockData.rtc_usec_divisor
#define rtclock_timebase_const                  RTClockData.rtc_timebase_const
#define rtclock_base_abstime                    RTClockData.rtc_base.abstime
#define rtclock_base_abstime_low                RTClockData.rtc_base.abstime_val.low
#define rtclock_base_abstime_high               RTClockData.rtc_base.abstime_val.high
#define rtclock_adj_abstime                             RTClockData.rtc_adj.abstime
#define rtclock_adj_abstime_low                 RTClockData.rtc_adj.abstime_val.low
#define rtclock_adj_abstime_high                RTClockData.rtc_adj.abstime_val.high
#define rtclock_timebase_func                   RTClockData.rtc_timebase_func

/* Only needed for AIC manipulation */
#define rtclock_timebase_addr                   RTClockData.rtc_timebase_addr
#define rtclock_timebase_val                    RTClockData.rtc_timebase_val

extern uint64_t arm_timer_slop_max;

extern void rtclock_intr(unsigned int);
extern boolean_t SetIdlePop(void);

extern void ClearIdlePop(boolean_t);
extern void rtclock_early_init(void);

#endif /* _ARM_RTCLOCK_H_ */
