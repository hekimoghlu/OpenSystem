/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
**  NAME:
**
**      uuidsys.c
**
**  FACILITY:
**
**      UUID
**
**  ABSTRACT:
**
**      UUID - Unix dependant (therefore system dependant) routines
**
**
*/

#ifndef UUID_BUILD_STANDALONE
#include <dce/dce.h>
#include <dce/uuid.h>           /* uuid idl definitions (public)        */
#include <dce/dce_utils.h>      /* defines  dce_get_802_addr()          */
#else
#include "uuid.h"
#endif
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "uuid_i.h"              /* uuid idl definitions (private)       */

#include <sys/time.h>           /* for struct timeval */
/*
 *  Define constant designation difference in Unix and DTSS base times:
 *  DTSS UTC base time is October 15, 1582.
 *  Unix base time is January 1, 1970.
 */
#define uuid_c_os_base_time_diff_lo     0x13814000
#define uuid_c_os_base_time_diff_hi     0x01B21DD2

/*
 * U U I D _ _ G E T _ O S _ T I M E
 *
 * Get OS time - contains platform-specific code.
 */
void uuid__get_os_time (uuid_time_t * uuid_time)
{

    struct timeval      tp;
    unsigned64_t        utc,
                        usecs,
                        os_basetime_diff;

    /*
     * Get current time
     */
    if (gettimeofday (&tp, (struct timezone *) 0))
    {
        perror ("uuid__get_os_time");
        exit (-1);
    }

    /*
     * Multiply the number of seconds by the number clunks
     */
    uuid__uemul ((unsigned32) tp.tv_sec, UUID_C_100NS_PER_SEC, &utc);

    /*
     * Multiply the number of microseconds by the number clunks
     * and add to the seconds
     */
    uuid__uemul ((unsigned32) tp.tv_usec, UUID_C_100NS_PER_USEC, &usecs);
    UADD_UVLW_2_UVLW (&usecs, &utc, &utc);

    /*
     * Offset between DTSS formatted times and Unix formatted times.
     */
    os_basetime_diff.lo = uuid_c_os_base_time_diff_lo;
    os_basetime_diff.hi = uuid_c_os_base_time_diff_hi;
    UADD_UVLW_2_UVLW (&utc, &os_basetime_diff, uuid_time);

}

/*
 * U U I D _ _ G E T _ O S _ P I D
 *
 * Get the process id
 */
unsigned32 uuid__get_os_pid ( void )
{
    return ((unsigned32) getpid());
}

/*
 * U U I D _ _ G E T _ O S _ A D D R E S S
 *
 * Wrapper for dce_get_802_addr()
 *
 * Kruntime has kernel specific version of this.
 */
void uuid__get_os_address
(
    uuid_address_p_t        addr,
    unsigned32              *status
)
{
    /*
     * Cheat and cast the uuid_address_p_t to dce_802_addr_t
     * since they are the same
     */

    dce_get_802_addr((dce_802_addr_t *)addr, status);

    if (*status == error_status_ok)
	*status = uuid_s_ok;
}
