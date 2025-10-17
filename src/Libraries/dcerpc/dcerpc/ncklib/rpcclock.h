/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
**
**  NAME:
**
**      rpcclock.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**
*/

#ifndef _RPCCLOCK_H
#define _RPCCLOCK_H	1

/*
 * Number of times per second the clock ticks (LSB weight for time values).
 */

#define  RPC_C_CLOCK_HZ             5

#define  RPC_CLOCK_SEC(sec)         ((sec)*RPC_C_CLOCK_HZ)
#define  RPC_CLOCK_MS(ms)           ((ms)/(1000/RPC_C_CLOCK_HZ))

typedef time_t  rpc_clock_t, *rpc_clock_p_t;

/*
 * An absolute time, UNIX time(2) format (i.e. time since 00:00:00 GMT,
 * Jan. 1, 1970, measured in seconds.
 */

typedef time_t  rpc_clock_unix_t, *rpc_clock_unix_p_t;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Get the current approximate tick count.  This routine is used to
 * timestamp data structures.  The tick count returned is only updated
 * by the rpc_timer routines once each time through the select listen
 * loop.  This degree of accuracy should be adequate for the purpose
 * of tracking the age of a data structure.
 */

PRIVATE rpc_clock_t rpc__clock_stamp (void);

/*
 * A routine to determine whether a specified time interval has passed.
 */
PRIVATE boolean rpc__clock_aged    (
        rpc_clock_t          /*time*/,
        rpc_clock_t          /*interval*/
    );

/*
 * Update the current tick count.  This routine is the only one that
 * actually makes system calls to obtain the time, and should only be
 * called from within the rpc_timer routines themselves.  Everyone else
 * should use the  routine rpc_timer_get_current_time which returns an
 * approximate tick count, or rpc_timer_aged which uses the approximate
 * tick count.  The value returned is the current tick count just
 * calculated.
 */
PRIVATE void rpc__clock_update ( void );

/*
 * Determine if a UNIX absolute time has expired
 * (relative to the system's current time).
 */

PRIVATE boolean rpc__clock_unix_expired (
        rpc_clock_unix_t    /*time*/
    );

/*
 * Convert an rpc_clock_t back to a "struct timespec".
 */

PRIVATE void rpc__clock_timespec (
	rpc_clock_t  /*clock*/,
	struct timespec * /*ts*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _RPCCLOCK_H */
