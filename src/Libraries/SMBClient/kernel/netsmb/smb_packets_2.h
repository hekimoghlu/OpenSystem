/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#ifndef _NETSMB_SMB_PACKETS_2_h
#define _NETSMB_SMB_PACKETS_2_h

#include <sys/syslog.h>

#define	SMB2_SIGNATURE			"\xFESMB"
#define	SMB2_SIGLEN				4
#define	SMB2_HDRLEN				64

typedef uint32_t DWORD;

typedef struct _FILETIME {
    DWORD dwLowDateTime;
    DWORD dwHighDateTime;
} FILETIME;

struct smb2_header
{
    uint8_t     protocol_id[4];
    uint16_t    structure_size;
    uint16_t    credit_charge;
    uint32_t    status; /* nt_status */
    uint16_t    command;
    uint16_t    credit_reqrsp;
    uint32_t    flags;
    uint32_t    next_command;
    uint64_t    message_id;
    
    union {
        /* Async commands have an async ID. */
        struct {
            uint64_t    async_id;
        } async;
        
        /* Sync command have a tree and process ID. */
        struct {
            uint32_t    process_id;
            uint32_t    tree_id;
        } sync;
    };
    
    uint64_t    session_id;
    uint8_t     signature[16];
    
    //enum { minimum_size = 64, maximum_size = 64 };
    //enum { defined_size = 64 };
};

#endif
