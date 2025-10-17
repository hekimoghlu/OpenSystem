/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
**  NAME
**
**      cnmgmt.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  The NCA Connection Protocol Service's Management Service.
**
**
*/

#include <commonp.h>    /* Common declarations for all RPC runtime */
#include <com.h>        /* Common communications services */
#include <comprot.h>    /* Common protocol services */
#include <cnp.h>        /* NCA Connection private declarations */
#include <cnmgmt.h>


/*
**++
**
**  ROUTINE NAME:       rpc__cn_mgmt_init
**
**  SCOPE:              PRIVATE - declared in cnmgmt.h,
**                                called from cninit.
**
**  DESCRIPTION:
**
**  Initialize the Connection management data collection
**  registers.
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   Management counters cleared.
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE void rpc__cn_mgmt_init (void)
{
    memset (&rpc_g_cn_mgmt, 0, sizeof (rpc_g_cn_mgmt));
}

/*
**++
**
**  Routine NAME:       rpc__cn_mgmt_inq_calls_sent
**
**  SCOPE:              PRIVATE - declared in cnmgmt.h
**
**  DESCRIPTION:
**
**  Report the total number of RPC that have been sent by
**  the NCA Connection Protocol.
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:
**
**      return          The number of RPCs sent through the NCA
**                      Connection Protocol Service.
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE unsigned32 rpc__cn_mgmt_inq_calls_sent (void)

{
    return (rpc_g_cn_mgmt.calls_sent);
}



/*
**++
**
**  Routine NAME:       rpc__cn_mgmt_inq_calls_rcvd
**
**  SCOPE:              PRIVATE - declared in cnmgmt.h
**
**  DESCRIPTION:
**
**  Report the total number of RPCs that have been received by
**  the NCA Connection Protocol.
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:
**
**      return          The number of RPCs received through the NCA
**                      Connection Protocol Service.
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE unsigned32 rpc__cn_mgmt_inq_calls_rcvd (void)

{
    return (rpc_g_cn_mgmt.calls_rcvd);
}


/*
**++
**
**  ROUTINE NAME:       rpc__cn_mgmt_inq_pkts_sent
**
**  SCOPE:              PRIVATE - declared in cnmgmt.h
**
**  DESCRIPTION:
**
**  Report the total number of packets that have been sent by
**  the NCA Connection Protocol.
**
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:
**
**      return          The number of RPC packets sent by the NCA
**                      Connection Protocol Service.
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE unsigned32 rpc__cn_mgmt_inq_pkts_sent (void)

{

    return (rpc_g_cn_mgmt.pkts_sent);
}


/*
**++
**
**  ROUTINE NAME:       rpc__cn_mgmt_inq_pkts_rcvd
**
**  SCOPE:              PRIVATE - declared in cnmgmt.h
**
**  DESCRIPTION:
**
**  Report the total number of packets that have been received by
**  the NCA Connection Protocol.
**
**  INPUTS:             none
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:
**
**      return          The number of RPC packets received by
**                      the NCA Connection Protocol Service.
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE unsigned32 rpc__cn_mgmt_inq_pkts_rcvd (void)
{
    return (rpc_g_cn_mgmt.pkts_rcvd);
}
