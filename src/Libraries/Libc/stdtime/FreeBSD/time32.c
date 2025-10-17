/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdtime/time32.c,v 1.5 2002/06/17 01:42:29 wollman Exp $");

#include <sys/types.h>
#include <timeconv.h>

/*
 * Convert a 32 bit representation of time_t into time_t.  XXX needs to
 * implement the 50-year rule to handle post-2038 conversions.
 */
time_t
_time32_to_time(__int32_t t32)
{
    return((time_t)t32);
}

/*
 * Convert time_t to a 32 bit representation.  If time_t is 64 bits we can
 * simply chop it down.   The resulting 32 bit representation can be 
 * converted back to a temporally local 64 bit time_t using time32_to_time.
 */
__int32_t
_time_to_time32(time_t t)
{
    return((__int32_t)t);
}

/*
 * Convert a 64 bit representation of time_t into time_t.  If time_t is
 * represented as 32 bits we can simply chop it and not support times
 * past 2038.
 */
time_t
_time64_to_time(__int64_t t64)
{
    return((time_t)t64);
}

/*
 * Convert time_t to a 64 bit representation.  If time_t is represented
 * as 32 bits we simply sign-extend and do not support times past 2038.
 */
__int64_t
_time_to_time64(time_t t)
{
    return((__int64_t)t);
}

/*
 * Convert to/from 'long'.  Depending on the sizeof(long) this may or 
 * may not require using the 50-year rule.
 */
long
_time_to_long(time_t t)
{
    if (sizeof(long) == sizeof(__int64_t))
	return(_time_to_time64(t));
    return((long)t);
}

time_t
_long_to_time(long tlong)
{
    if (sizeof(long) == sizeof(__int32_t))
	return(_time32_to_time(tlong));
    return((time_t)tlong);
}

/*
 * Convert to/from 'int'.  Depending on the sizeof(int) this may or 
 * may not require using the 50-year rule.
 */
int
_time_to_int(time_t t)
{
    if (sizeof(int) == sizeof(__int64_t))
	return(_time_to_time64(t));
    return((int)t);
}

time_t
_int_to_time(int tint)
{
    if (sizeof(int) == sizeof(__int32_t))
	return(_time32_to_time(tint));
    return((time_t)tint);
}
