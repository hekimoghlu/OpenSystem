/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#ifndef _OSX_NTFS_TIME_H
#define _OSX_NTFS_TIME_H

#include <sys/time.h>

#include "ntfs_endian.h"
#include "ntfs_types.h"

#define NTFS_TIME_OFFSET ((s64)(369 * 365 + 89) * 24 * 3600 * 10000000)

/**
 * utc2ntfs - convert OS X time to NTFS time
 * @ts:		OS X UTC time to convert to NTFS (little endian) time
 *
 * Convert the OS X UTC time @ts to its corresponding NTFS time and return that
 * in little endian format.
 *
 * OS X stores time in a struct timespec consisting of a time_t (long at
 * present) tv_sec and a long tv_nsec where tv_sec is the number of 1-second
 * intervals since 1st January 1970, 00:00:00 UTC and tv_nsec is the number of
 * 1-nano-second intervals since the value of tv_sec.
 *
 * NTFS uses Microsoft's standard time format which is stored in a s64 and is
 * measured as the number of 100-nano-second intervals since 1st January 1601,
 * 00:00:00 UTC.
 */
static inline sle64 utc2ntfs(const struct timespec ts)
{
	/*
	 * Convert the seconds to 100ns intervals, add the nano-seconds
	 * converted to 100ns intervals, and then add the NTFS time offset.
	 */
	return cpu_to_sle64((s64)ts.tv_sec * 10000000 + ts.tv_nsec / 100 +
			NTFS_TIME_OFFSET);
}

/**
 * ntfs_utc_current_time - get the current time in OS X time
 *
 * Get the current time from the OS X kernel, round it down to the nearest
 * 100-nano-second interval and return that in cpu format.
 */
static inline struct timespec ntfs_utc_current_time(void)
{
	struct timespec ts;

	nanotime(&ts);
	/* Round down to nearest 100-nano-second interval. */
	ts.tv_nsec -= ts.tv_nsec % 100;
	return ts;
}

/**
 * ntfs_current_time - get the current time in little endian NTFS format
 *
 * Get the current time from the OS X kernel, convert it to its corresponding
 * NTFS time and return that in little endian format.
 */
static inline sle64 ntfs_current_time(void)
{
	struct timespec ts;

	nanotime(&ts);
	return utc2ntfs(ts);
}

/**
 * ntfs2utc - convert NTFS time to OS X time
 * @time:	NTFS time (little endian) to convert to OS X UTC
 *
 * Convert the little endian NTFS time @time to its corresponding OS X UTC time
 * and return that in cpu format.
 *
 * OS X stores time in a struct timespec consisting of a time_t (long at
 * present) tv_sec and a long tv_nsec where tv_sec is the number of 1-second
 * intervals since 1st January 1970, 00:00:00 UTC without including leap
 * seconds and tv_nsec is the number of 1-nano-second intervals since the value
 * of tv_sec.
 *
 * NTFS uses Microsoft's standard time format which is stored in a s64 and is
 * measured as the number of 100 nano-second intervals since 1st January 1601,
 * 00:00:00 UTC.
 *
 * FIXME: There does not appear to be an asm optimized function in xnu to do
 * the division and return the remainder in one single step.  If there is or
 * one gets added at some point the below division and remainder determination
 * should be combined into a single step using it.
 */
static inline struct timespec ntfs2utc(const sle64 time)
{
	u64 t;
	struct timespec ts;

	/* Subtract the NTFS time offset. */
	t = (u64)(sle64_to_cpu(time) - NTFS_TIME_OFFSET);
	/*
	 * Convert the time to 1-second intervals and the remainder to
	 * 1-nano-second intervals.
	 */
	ts.tv_sec = t / 10000000;
	ts.tv_nsec = (t % 10000000) * 100;
	return ts;
}

#endif /* _OSX_NTFS_TIME_H */
