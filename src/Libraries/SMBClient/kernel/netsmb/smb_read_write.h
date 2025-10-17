/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#ifndef smb_read_write_h
#define smb_read_write_h

#define SMB_MAX_RW_HASH_SZ    12    /* Number of global worker threads */
#define SMB_STRATEGY_HASH_SZ   4    /* Number of strategy worker threads */

void smb_rw_init(void);
void smb_rw_cleanup(void);
void smb_rw_proxy(void *arg);

/* smb_rw_arg commands */
typedef enum _SMB_RW_CMD_FLAGS
{
    SMB_READ_WRITE = 0x0001,         /* Read/write */
    SMB_LEASE_BREAK_ACK = 0x0002,    /* Lease break ack exchange */
    SMB_VNOP_STRATEGY = 0x0004,      /* vnop_strategy read/write */
} _SMB_RW_CMD_FLAGS;

/* smb_rw_arg flags */
typedef enum _SMB_RW_ARG_FLAGS
{
    SMB_RW_QUEUED = 0x0001,         /* enqueued and waiting to be sent */
    SMB_RW_IN_USE = 0x0002,         /* this pb is currently being used */
    SMB_RW_REPLY_RCVD = 0x0004      /* reply has arrived */
} _SMB_RW_ARG_FLAGS;

/* smb_RW_FLAGS flags */
typedef enum _SMB_RW_FLAGS
{
    SMB_RW_ANY_QUEUE_ID =         0x0001,
    SMB_RW_USE_QUEUE_ID =         0x0002,
    SMB_RW_AVOID_QUEUE_ID =       0x0004
} _SMB_RW_FLAGS;

struct smb_rw_arg {
    /* Common */
    TAILQ_ENTRY(smb_rw_arg) sra_svcq;
    
    uint32_t command;
    uint32_t flags;
    lck_mtx_t rw_arg_lock;
    int error;

    union {
        /* Read/write */
        struct {
            struct smb2_rw_rq *read_writep;
            struct smb_rq *rqp;
            user_ssize_t resid;
        } rw;
        
        /* Lease Break Ack */
        struct {
            struct smb_share *share;
            struct smbiod *iod;
            uint64_t lease_key_hi;
            uint64_t lease_key_low;
            uint32_t lease_state;
            uint32_t ret_lease_state;
            vfs_context_t context;
        } lease;
        
        struct {
            struct buf *bp;
        } strategy;
    };
};


#endif /* smb_read_write_h */
