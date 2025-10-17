/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
**      rpcdutil.c
**
**  FACILITY:
**
**      RPC Daemon Utility Routines
**
**  ABSTRACT:
**
**  RPC Daemon Utility Routines - protocol tower manipulation, sleep primitives
**
**
*/

#include <commonp.h>
#include <com.h>

#include <dce/ep.h>     /* derived from ep.idl */
#include <dsm.h>        /* derived from dsm.idl */

#include <rpcdp.h>
#include <rpcddb.h>
#include <rpcdepdbp.h>
#include <rpcdutil.h>

#include <comtwr.h>
#include <comtwrflr.h>
#include <comtwrref.h>

#ifdef DEBUG

static void print_bad_tower(
	twr_p_t         tower,
	const char      *file,
	int             line)
{
    unsigned32 i;

    fprintf(stderr, "Bad tower (%s, line %d); length=%lu\n    octets:\n",
            file, line, (unsigned long) tower->tower_length);

    for (i = 0; i < tower->tower_length; i++)
        fprintf(stderr, "%02x", tower->tower_octet_string[i]);

    fprintf(stderr, "\n");
}

#define CHECK_TOWER_STATUS(tower, status) \
    assert(status != NULL); \
    if (dflag && *(status) != rpc_s_ok) \
        print_bad_tower((tower), __FILE__, __LINE__);

#else

#define CHECK_TOWER_STATUS(tower, status)

#endif

/*  Parse and check a tower
 *  Fill entp's fields derived from the tower
 */
PRIVATE void tower_to_fields(tower, tfp, status)
twr_p_t         tower;
twr_fields_t    *tfp;
error_status_t  *status;
{
    rpc_tower_ref_t *tref;
    error_status_t  tmp_st;

    rpc__tower_to_tower_ref(tower, &tref, status);
    if (! STATUS_OK(status)) return;

    if (tref->count < (unsigned16)(RPC_C_NUM_RPC_FLOORS + 1))
    {
        assert(status != NULL);
        SET_STATUS(status, ept_s_invalid_entry);
    }

    if (STATUS_OK(status))
        rpc__tower_ref_inq_protseq_id(tref, &tfp->protseq, status);

    if (STATUS_OK(status))
        rpc__tower_flr_to_if_id(tref->floor[0], &tfp->interface, status);

    if (STATUS_OK(status))
        rpc__tower_flr_to_drep(tref->floor[1], &tfp->data_rep, status);

    if (STATUS_OK(status))
        rpc__tower_flr_to_rpc_prot_id(tref->floor[2],
            &tfp->rpc_protocol, &tfp->rpc_protocol_vers_major,
            &tfp->rpc_protocol_vers_minor, status);

    CHECK_TOWER_STATUS(tower, status);

    rpc__tower_ref_free(&tref, &tmp_st);
}

PRIVATE void tower_to_addr(tower, addr, status)
twr_p_t         tower;
rpc_addr_p_t    *addr;
error_status_t  *status;
{
    *addr = NULL;

    rpc__naf_tower_flrs_to_addr(tower->tower_octet_string, addr, status);

    CHECK_TOWER_STATUS(tower, status);
}

PRIVATE void tower_to_if_id(tower, if_id, status)
twr_p_t         tower;
rpc_if_id_t     *if_id;
error_status_t  *status;
{
    rpc_tower_ref_t *tref;
    error_status_t  tmp_st;

    rpc__tower_to_tower_ref(tower, &tref, status);
    if (! STATUS_OK(status)) goto DONE;

    if (tref->count < (unsigned16)RPC_C_NUM_RPC_FLOORS)
    {
        assert(status != NULL);
        SET_STATUS(status, ept_s_invalid_entry);
        goto DONE;
    }

    rpc__tower_flr_to_if_id(tref->floor[0], if_id, status);

    rpc__tower_ref_free(&tref, &tmp_st);

DONE:

    CHECK_TOWER_STATUS(tower, status);

}

PRIVATE void tower_ss_copy(src_tower, dest_tower, status)
twr_p_t         src_tower;
twr_p_t         *dest_tower;
error_status_t  *status;
{
    twr_p_t     dtp;

    *dest_tower = (twr_p_t) rpc_ss_allocate(sizeof(twr_t) + (src_tower->tower_length - 1));
    if (*dest_tower == NULL)
    {
        SET_STATUS(status, ept_s_no_memory);
        return;
    }

    dtp = *dest_tower;
    dtp->tower_length = src_tower->tower_length;
    memcpy((char *) dtp->tower_octet_string, (char *) src_tower->tower_octet_string,
        src_tower->tower_length);

    SET_STATUS_OK(status);
}


/*  Sleep until starttime + nsecs
 */
PRIVATE void ru_sleep_until(starttime, nsecs)
struct timeval  *starttime;
unsigned32      nsecs;
{
    unsigned32      waketime;
    struct timeval  now;
    struct timezone tz;
    unsigned32      sleep_secs;

    waketime = starttime->tv_sec + nsecs;
    gettimeofday(&now, &tz);
    if (waketime > (unsigned32)now.tv_sec)
    {
        sleep_secs = waketime - now.tv_sec;
        ru_sleep(sleep_secs);
    }
}

/*  Sleep for nsecs
 */
PRIVATE void ru_sleep(nsecs)
unsigned32      nsecs;
{
    struct timespec  sleeptime;
    sleeptime.tv_sec = nsecs;
    sleeptime.tv_nsec = 0;
    dcethread_delay(&sleeptime);
}
