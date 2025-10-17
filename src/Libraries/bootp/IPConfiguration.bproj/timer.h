/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
 * timer_private.h
 * - timer functions
 */

/* 
 * Modification History
 *
 * May 8, 2000	Dieter Siegmund (dieter@apple)
 * - created
 */

#ifndef _S_TIMER_H
#define _S_TIMER_H

#include <mach/boolean.h>
#include "dynarray.h"
#include "symbol_scope.h"
#include <dispatch/dispatch.h>
#include <CoreFoundation/CFDate.h>

typedef CFAbsoluteTime absolute_time_t;

typedef struct timer_callout timer_callout_t;

typedef void (timer_func_t)(void * arg1, void * arg2, void * arg3);

absolute_time_t		timer_get_current_time();

INLINE absolute_time_t
timer_current_secs(void)
{
	return timer_get_current_time();
}

/**
 ** callout functions
 **/
timer_callout_t *	timer_callout_init(const char * name);
void			timer_callout_free(timer_callout_t * * callout_p);

int			timer_set_relative(timer_callout_t * entry,
					   struct timeval rel_time,
					   timer_func_t * func,
					   void * arg1, void * arg2,
					   void * arg3);
int			timer_callout_set(timer_callout_t * callout,
					  absolute_time_t relative_time,
					  timer_func_t * func,
					  void * arg1, void * arg2,
					  void * arg3);
int			timer_callout_set_absolute(timer_callout_t * callout,
						   absolute_time_t wakeup_time,
						   timer_func_t * func,
						   void * arg1, void * arg2,
						   void * arg3);
void			timer_cancel(timer_callout_t * entry);

boolean_t		timer_time_changed(timer_callout_t * entry);

boolean_t		timer_still_pending(timer_callout_t * entry);

#endif /* _S_TIMER_H */
