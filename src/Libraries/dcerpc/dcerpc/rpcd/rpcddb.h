/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
**      rpcddb.h
**
**  FACILITY:
**
**      RPC Daemon Basic Database Routines - header file
**
**  ABSTRACT:
**
**  RPC Daemon basic database "private" types, defines, routines, ...
**
**
*/

#ifndef RPCDDB_H
#define RPCDDB_H

/*
 *  Database entry list and hash table management typedefs
 */

typedef enum db_list_type_t
{
    db_c_entry_list, db_c_object_list, db_c_interface_list
} db_list_type_t;

/*
 *  forward and backward ptrs when used within an entry.
 *  first and last entry on list when used to manage a list
 */

typedef struct db_list
{
    struct db_list      *fwd;
    struct db_list      *back;
} db_list_t;

typedef struct
{
    db_list_t       entry_list;
    db_list_t       object_list;
    db_list_t       interface_list;
} db_lists_t, *db_lists_p_t;

#define db_c_nbucket  64

typedef db_list_t   db_hash_table_t[db_c_nbucket];

typedef struct
{
    db_list_t           entry_list;
    db_hash_table_t     object_table;
    db_hash_table_t     interface_table;
} db_lists_mgmt_t;

/*
 * The header of a persistant endpoint database file.
 */
struct db_file_hdr {
    unsigned32  version;
    idl_uuid_t      object;
};

/*
 * The internal structure of an ep database handle.
 */
struct db {
    dsm_handle_t        dsh;
    idl_uuid_t              object;
    db_lists_mgmt_t     lists_mgmt;         /* entry, object, interface lists mgmt */
    dcethread_mutex     lock;               /* Database mutex lock */
    dcethread*          sliv_task1_h;       /* Server liveness task 1 handle */
    dcethread*          sliv_task2_h;       /* Server liveness task 2 handle */
    dcethread_cond      sliv_task2_cv;      /* Server liveness condition variable used by
                                               sliv_task1 to send an event to sliv_task2 */
};

/*
 * This internal entry format is required to support
 * the forwarding map mechanism (hence the rpc addr)
 * and the DCE 1.0 endpoint mapper network interface.
 * The queues are recreated at startup time.  The addr
 * field is derived from the tower at startup time.
 * All other fields are stablely stored on the disk.
 */

typedef struct
{
    db_lists_t              lists;
    unsigned16              read_nrefs;     /* # readers of this entry who have given up db_lock */
    unsigned16              ncomm_fails;    /* # consecutive pings that have failed to this server */
    ndr_boolean             delete_flag;    /* entry should be deleted when read_nrefs == 0 */
    ndr_boolean             object_nil;
    ndr_boolean             if_nil;
    idl_uuid_t                  object;
    rpc_if_id_t             interface;
    idl_uuid_t                  data_rep_id;
    unsigned32              data_rep_vers_major;
    unsigned32              data_rep_vers_minor;
    rpc_protocol_id_t       rpc_protocol;
    unsigned32              rpc_protocol_vers_major;
    unsigned32              rpc_protocol_vers_minor;
    idl_uuid_t                  type;           /* for LB compat */
    unsigned32              llb_flags;      /* for LB compat */
    unsigned32              saddr_len;      /* for LB compat */
    rpc_addr_p_t            addr;
    unsigned_char_t         annotation[ept_max_annotation_size];
    twr_t                   tower;
} db_entry_t, *db_entry_p_t;

/*
 *  Max read references to an entry (must fit epdb_entry_t.read_nrefs - unsigned16)
 *  Leave some room for sliv_task1 and sliv_task2 so they don't need to check
 *  for overflow.
 */
#define db_c_max_read_nrefs     0x0fff0

PRIVATE void db_open
    (
        struct db           *h,
        unsigned char       *database_file,
        unsigned32          version,
        error_status_t      *status
    );

/*
 * Update entry in place on disk
 */
PRIVATE void db_update_entry
    (
        struct db       *h,
        db_entry_p_t    entp,
        error_status_t  *status
    );

PRIVATE void db_init_lists
    (
        struct db           *h
    );

PRIVATE void db_lists_add
    (
        struct db       *h,
        db_entry_t      *entp
    );

PRIVATE void db_lists_remove
    (
        struct db       *h,
        db_entry_t      *entp
    );

PRIVATE void db_htable_add
    (
        db_hash_table_t     htable,
        db_list_type_t      table_type,
        uuid_p_t            id,
        db_lists_t          *entp
    );

PRIVATE void db_htable_remove
    (
        db_hash_table_t     htable,
        db_list_type_t      table_type,
        uuid_p_t            id,
        db_lists_t          *entp
    );

PRIVATE void db_list_add
    (
        db_list_t           *list,
        db_list_type_t      list_type,
        db_lists_t          *entp
    );

PRIVATE void db_list_remove
    (
        db_list_t           *list,
        db_list_type_t      list_type,
        db_lists_t          *entp
    );

PRIVATE db_lists_t *db_list_first
    (
        db_lists_mgmt_t     *lists_mgmt,
        db_list_type_t      list_type,
        uuid_p_t            id
    );

PRIVATE db_lists_t *db_list_next
    (
        db_list_type_t      list_type,
        db_lists_t          *entp
    );

PRIVATE void db_save_context
    (
        struct db           *h,
        ept_lookup_handle_t *entry_handle,
        db_list_type_t      list_type,
        db_lists_t          *lp,
        unsigned32          pass
    );

PRIVATE void db_delete_context
    (
        struct db           *h,
        ept_lookup_handle_t *entry_handle
    );

PRIVATE void db_get_context
    (
        struct db           *h,
        ept_lookup_handle_t *entry_handle,
        db_list_type_t      *list_type,
        db_lists_t          **lp,
        unsigned32          *pass,
        error_status_t      *status
    );

PRIVATE boolean32 db_different_context
    (
        struct db           *h,
        ept_lookup_handle_t *entry_handle,
        error_status_t      *status
    );

PRIVATE void db_lock
    (
        struct db *h
    );

PRIVATE void db_unlock
    (
        struct db *h
    );

PRIVATE void db_init_lock
    (
        struct db *h
    );

PRIVATE void db_to_ept_ecode
    (
        error_status_t      *status
    );

#endif
