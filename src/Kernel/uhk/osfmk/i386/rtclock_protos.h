/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
/*
 *	File:		rtclock_protos.h
 *	Purpose:	C Routines for handling the machine dependent
 *				real-time clock.
 */

#ifndef _I386_RTCLOCK_PROTOS_H_
#define _I386_RTCLOCK_PROTOS_H_

typedef struct pal_rtc_nanotime pal_rtc_nanotime_t;
extern uint64_t tsc_rebase_abs_time;

extern void     _rtc_nanotime_adjust(
	uint64_t                tsc_base_delta,
	pal_rtc_nanotime_t      *dst);

extern uint64_t _rtc_nanotime_read(
	pal_rtc_nanotime_t      *rntp);

extern uint64_t _rtc_tsc_to_nanoseconds(
	uint64_t    value,
	pal_rtc_nanotime_t      *rntp);

extern int     rtclock_intr(x86_saved_state_t *regs);


/*
 * Timer control.
 */
typedef struct {
	void     (*rtc_config)(void);
	uint64_t (*rtc_set)(uint64_t, uint64_t);
} rtc_timer_t;
extern rtc_timer_t      *rtc_timer;

extern void             rtc_timer_init(void);

extern void             rtclock_early_init(void);
extern void             rtc_nanotime_init(uint64_t);
extern void             rtc_decrementer_configure(void);
#endif /* _I386_RTCLOCK_PROTOS_H_ */
