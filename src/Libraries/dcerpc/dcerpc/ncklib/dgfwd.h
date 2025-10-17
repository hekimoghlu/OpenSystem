/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
**      dgfwd.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG Packet Forwarding handler
**
**
*/

#ifndef _DGFWD_H
#define _DGFWD_H

/*
 * R P C _ _ D G _ F W D _ P K T
 *
 * Forwarding Service.
 */

PRIVATE unsigned32 rpc__dg_fwd_pkt    (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

/*
 * Can return three values:
 *     FWD_PKT_NOTDONE  - caller should handle packet
 *     FWD_PKT_DONE     - we handled the packet, ok to free it
 *     FWD_PKT_DELAYED  - we saved it, don't handle it, don't free it.
 */
#define	FWD_PKT_NOTDONE		0
#define FWD_PKT_DONE		1
#define FWD_PKT_DELAYED		2

/*
 * R P C _ _ D G _ F W D _ I N I T
 *
 * Initialize forwarding service private mutex.
 */

PRIVATE void rpc__dg_fwd_init (void);

#endif /* _DGFWD_H */
