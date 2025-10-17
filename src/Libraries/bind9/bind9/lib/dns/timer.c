/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
/* $Id: timer.c,v 1.7 2007/06/19 23:47:16 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/result.h>
#include <isc/time.h>
#include <isc/timer.h>

#include <dns/types.h>
#include <dns/timer.h>

#define CHECK(op) \
	do { result = (op);					\
		if (result != ISC_R_SUCCESS) goto failure;	\
	} while (0)

isc_result_t
dns_timer_setidle(isc_timer_t *timer, unsigned int maxtime,
		  unsigned int idletime, isc_boolean_t purge)
{
	isc_result_t result;
	isc_interval_t maxinterval, idleinterval;
	isc_time_t expires;

	/* Compute the time of expiry. */
	isc_interval_set(&maxinterval, maxtime, 0);
	CHECK(isc_time_nowplusinterval(&expires, &maxinterval));

	/*
	 * Compute the idle interval, and add a spare nanosecond to
	 * work around the silly limitation of the ISC timer interface
	 * that you cannot specify an idle interval of zero.
	 */
	isc_interval_set(&idleinterval, idletime, 1);

	CHECK(isc_timer_reset(timer, isc_timertype_once,
			      &expires, &idleinterval,
			      purge));
 failure:
	return (result);
}
