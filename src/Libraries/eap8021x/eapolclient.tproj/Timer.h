/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
 * Timer.h
 * - timer functions
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple)
 * - created (from bootp/IPConfiguration/timer.h)
 */

#ifndef _S_TIMER_H
#define _S_TIMER_H
#include <sys/time.h>

typedef struct Timer_s Timer, *TimerRef;

typedef void (Timer_func_t)(void * arg1, void * arg2, void * arg3);

struct timeval		Timer_current_time(void);
long			Timer_current_secs(void);
TimerRef		Timer_create(void);
void			Timer_free(TimerRef * callout_p);
int			Timer_set_relative(TimerRef entry,
					   struct timeval rel_time,
					   Timer_func_t * func,
					   void * arg1, void * arg2,
					   void * arg3);
void			Timer_cancel(TimerRef entry);

#endif /* _S_TIMER_H */

