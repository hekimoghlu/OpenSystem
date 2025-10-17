/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
**      comtwrflr.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Contains private definitions and prototypes of the
**      comtwrflr.c module.
**
**
*/

#ifndef _COMTWRFLR_H
#define _COMTWRFLR_H 1

/*
 * Constants
 */

/*
 * The architecturally defined tower floor protocol identifier
 * prefix to signify the succeeding data as an RPC uuid.
 */
#define RPC_C_PROT_ID_PREFIX    (0x0D)

/*
 * Prototypes
 */

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__tower_flr_free (
    rpc_tower_floor_p_t     * /*floor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_from_drep (
    rpc_syntax_id_p_t        /*transfer_syntax*/,
    rpc_tower_floor_p_t     * /*floor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_from_if_id (
    rpc_if_id_p_t            /*if_id*/,
    rpc_tower_floor_p_t     * /*floor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_from_rpc_prot_id (
    rpc_protseq_id_t         /*rpc_protseq_id*/,
    rpc_protocol_version_p_t /*protocol_version*/,
    rpc_tower_floor_p_t     * /*floor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_from_uuid (
    uuid_p_t                 /*uuid*/,
    unsigned32               /*version_major*/,
    unsigned32               /*version_minor*/,
    rpc_tower_floor_p_t     * /*floor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_id_from_uuid (
    uuid_p_t         /*uuid*/,
    unsigned32       /*version_major*/,
    unsigned32      * /*prot_id_len*/,
    unsigned8       ** /*prot_id*/,
    unsigned32      * /*status*/
);

PRIVATE void rpc__tower_flr_id_to_uuid (
    unsigned8       * /*prot_id*/,
    idl_uuid_t          * /*uuid*/,
    unsigned32      * /*version_major*/,
    unsigned32      * /*status*/
);

PRIVATE void rpc__tower_flr_to_drep (
    rpc_tower_floor_p_t      /*floor*/,
    rpc_syntax_id_t         * /*transfer_syntax*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_to_if_id (
    rpc_tower_floor_p_t      /*floor*/,
    rpc_if_id_t             * /*if_id*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_to_rpc_prot_id (
    rpc_tower_floor_p_t      /*floor*/,
    rpc_protocol_id_t       * /*rpc_protocol_id*/,
    unsigned32              * /*version_major*/,
    unsigned32              * /*version_minor*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_flr_to_uuid (
    rpc_tower_floor_p_t      /*floor*/,
    idl_uuid_t                  * /*uuid*/,
    unsigned32              * /*version_major*/,
    unsigned32              * /*version_minor*/,
    unsigned32              * /*status*/
);

#endif /* _COMTWRFLR_H */
