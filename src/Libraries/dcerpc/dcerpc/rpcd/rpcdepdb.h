/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
**      rpcdepdb.h
**
**  FACILITY:
**
**      RPC Daemon
**
**  ABSTRACT:
**
**  Generic Endpoint Database Manager.
**
**
*/

#ifndef RPCDEPDB_H
#define RPCDEPDB_H

typedef void *epdb_handle_t;

/*  Get the handle for the ep database from
 *  a handle to the endpoint object
 */
PRIVATE void epdb_handle_from_ohandle
    (
        handle_t            h,
        epdb_handle_t       *epdb_h,
        error_status_t      *status
    );

/*  Return the handle to the ep database
 */
PRIVATE epdb_handle_t epdb_inq_handle (void);

PRIVATE epdb_handle_t epdb_init
    (
        unsigned char       *pathname,
        error_status_t      *status
    );

PRIVATE void epdb_insert
    (
        epdb_handle_t       h,
        ept_entry_p_t       xentry,
        boolean32           replace,
        error_status_t      *status
    );

PRIVATE void epdb_delete
    (
        epdb_handle_t       h,
        ept_entry_p_t       xentry,
        error_status_t      *status
    );

PRIVATE void epdb_mgmt_delete
    (
        epdb_handle_t       h,
        boolean32           object_speced,
        uuid_p_t            object,
        twr_p_t             tower,
        error_status_t      *status
    );

PRIVATE void epdb_lookup
    (
        epdb_handle_t       h,
        unsigned32          inquiry_type,
        uuid_p_t            object,
        rpc_if_id_p_t       interface,
        unsigned32          vers_option,
        ept_lookup_handle_t *entry_handle,
        unsigned32          max_ents,
        unsigned32          *num_ents,
        ept_entry_t         entries[],
        error_status_t      *status
    );

PRIVATE void epdb_map
    (
        epdb_handle_t       h,
        uuid_p_t            object,
        twr_p_t             map_tower,
        ept_lookup_handle_t *entry_handle,
        unsigned32          max_towers,
        unsigned32          *num_towers,
        twr_t               *fwd_towers[],
        unsigned32          *status
    );

PRIVATE void epdb_fwd
    (
        epdb_handle_t       h,
        uuid_p_t            object,
        rpc_if_id_p_t       interface,
        rpc_syntax_id_p_t   data_rep,
        rpc_protocol_id_t   rpc_protocol,
        unsigned32          rpc_protocol_vers_major,
        unsigned32          rpc_protocol_vers_minor,
        rpc_addr_p_t        addr,
        ept_lookup_handle_t *map_handle,
        unsigned32          max_addrs,
        unsigned32          *num_addrs,
        rpc_addr_p_t        fwd_addrs[],
        unsigned32          *status
    );

PRIVATE void epdb_inq_object
    (
        epdb_handle_t h,
        idl_uuid_t *object,
        error_status_t *status
    );

PRIVATE void epdb_delete_lookup_handle
    (
        epdb_handle_t       h,
        ept_lookup_handle_t *entry_handle
    );

#endif
