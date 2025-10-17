/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
/* $Id: condition.c,v 1.36 2007/06/19 23:47:18 tbox Exp $ */

/*! \file */

#include <config.h>

#include <errno.h>

#include <isc/condition.h>
#include <isc/msgs.h>
#include <isc/strerror.h>
#include <isc/string.h>
#include <isc/time.h>
#include <isc/util.h>

isc_result_t
isc_condition_waituntil(isc_condition_t *c, isc_mutex_t *m, isc_time_t *t) {
	int presult;
	isc_result_t result;
	struct timespec ts;
	char strbuf[ISC_STRERRORSIZE];

	REQUIRE(c != NULL && m != NULL && t != NULL);

	/*
	 * POSIX defines a timespec's tv_sec as time_t.
	 */
	result = isc_time_secondsastimet(t, &ts.tv_sec);

	/*
	 * If we have a range error ts.tv_sec is most probably a signed
	 * 32 bit value.  Set ts.tv_sec to INT_MAX.  This is a kludge.
	 */
	if (result == ISC_R_RANGE)
		ts.tv_sec = INT_MAX;
	else if (result != ISC_R_SUCCESS)
		return (result);

	/*!
	 * POSIX defines a timespec's tv_nsec as long.  isc_time_nanoseconds
	 * ensures its return value is < 1 billion, which will fit in a long.
	 */
	ts.tv_nsec = (long)isc_time_nanoseconds(t);

	do {
#if ISC_MUTEX_PROFILE
		presult = pthread_cond_timedwait(c, &m->mutex, &ts);
#else
		presult = pthread_cond_timedwait(c, m, &ts);
#endif
		if (presult == 0)
			return (ISC_R_SUCCESS);
		if (presult == ETIMEDOUT)
			return (ISC_R_TIMEDOUT);
	} while (presult == EINTR);

	isc__strerror(presult, strbuf, sizeof(strbuf));
	UNEXPECTED_ERROR(__FILE__, __LINE__,
			 "pthread_cond_timedwait() %s %s",
			 isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
					ISC_MSG_RETURNED, "returned"),
			 strbuf);
	return (ISC_R_UNEXPECTED);
}
