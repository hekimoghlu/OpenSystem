/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
**
**  NAME:
**
**      nsrttwr.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Header file containing macros, definitions, typedefs and prototypes of
**      exported routines from the nsrttwr.c module.
**
**
**/

#ifndef _COMTWRREF_H
#define _COMTWRREF_H

/*
 * Type Definitions
 */

/*
 * Prototypes
 */

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__tower_ref_add_floor (
    unsigned32           /*floor_number*/,
    rpc_tower_floor_p_t  /*floor*/,
    rpc_tower_ref_t     * /*tower_ref*/,
    unsigned32          * /*status*/
);

PRIVATE void rpc__tower_ref_alloc (
    unsigned8           * /*tower_octet_string*/,
    unsigned32           /*num_flrs*/,
    unsigned32           /*start_flr*/,
    rpc_tower_ref_p_t   * /*tower_ref*/,
    unsigned32          * /*status*/
);

PRIVATE void rpc__tower_ref_copy (
    rpc_tower_ref_p_t    /*source_tower*/,
    rpc_tower_ref_p_t   * /*dest_tower*/,
    unsigned32          * /*status*/
);

PRIVATE void rpc__tower_ref_free (
    rpc_tower_ref_p_t       * /*tower_ref*/,
    unsigned32              * /*status*/
);

PRIVATE void rpc__tower_ref_inq_protseq_id (
    rpc_tower_ref_p_t    /*tower_ref*/,
    rpc_protseq_id_t    * /*protseq_id*/,
    unsigned32          * /*status*/
);

#if 0
/* Removed unused symbol for rdar://problem/26430747 */
PRIVATE boolean rpc__tower_ref_is_compatible (
    rpc_if_rep_p_t           /*if_spec*/,
    rpc_tower_ref_p_t        /*tower_ref*/,
    unsigned32              * /*status*/
);
#endif

PRIVATE void rpc__tower_ref_vec_free (
    rpc_tower_ref_vector_p_t    * /*tower_vector*/,
    unsigned32                  * /*status*/
);

PRIVATE void rpc__tower_ref_vec_from_binding (
    rpc_if_rep_p_t               /*if_spec*/,
    rpc_binding_handle_t         /*binding*/,
    rpc_tower_ref_vector_p_t    * /*tower_vector*/,
    unsigned32                  * /*status*/
);

#endif /* _COMTWRREF_H */
