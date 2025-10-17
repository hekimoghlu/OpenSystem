/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
**      rpcdutil.h
**
**  FACILITY:
**
**      RPC Daemon Utility Routines - header file
**
**  ABSTRACT:
**
**  RPC Daemon Utility Routines - protocol tower manipulation, sleep primitives
**
**
*/

#ifndef RPCDUTIL_H
#define RPCDUTIL_H

typedef struct
{
    rpc_if_id_t             interface;
    rpc_syntax_id_t         data_rep;
    rpc_protocol_id_t       rpc_protocol;
    unsigned32              rpc_protocol_vers_major;
    unsigned32              rpc_protocol_vers_minor;
    rpc_protseq_id_t        protseq;
} twr_fields_t, *twr_fields_p_t;

PRIVATE void tower_to_fields
    (
        twr_p_t         tower,
        twr_fields_t    *tfp,
        error_status_t  *status
    );

PRIVATE void tower_to_addr
    (
        twr_p_t         tower,
        rpc_addr_p_t    *addr,
        error_status_t  *status
    );

PRIVATE void tower_to_if_id
    (
        twr_p_t         tower,
        rpc_if_id_t     *if_id,
        error_status_t  *status
    );

PRIVATE void tower_ss_copy
    (
        twr_p_t         src_tower,
        twr_p_t         *dest_tower,
        error_status_t  *status
    );

PRIVATE void ru_sleep_until
    (
        struct timeval  *starttime,
        unsigned32      nsecs
    );

PRIVATE void ru_sleep
    (
        unsigned32      nsecs
    );

#endif
