/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
/* $Id: timer.h,v 1.9 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_TIMER_H
#define DNS_TIMER_H 1

/*! \file dns/timer.h */

/***
 ***	Imports
 ***/

#include <isc/buffer.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

/***
 ***	Functions
 ***/

isc_result_t
dns_timer_setidle(isc_timer_t *timer, unsigned int maxtime,
		  unsigned int idletime, isc_boolean_t purge);
/*%<
 * Convenience function for setting up simple, one-second-granularity
 * idle timers as used by zone transfers.
 * \brief
 * Set the timer 'timer' to go off after 'idletime' seconds of inactivity,
 * or after 'maxtime' at the very latest.  Events are purged iff
 * 'purge' is ISC_TRUE.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_TIMER_H */
