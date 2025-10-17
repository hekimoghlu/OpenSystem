/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef _KERN_BLOCK_HINT_H_
#define _KERN_BLOCK_HINT_H_

typedef enum thread_snapshot_wait_flags {
	kThreadWaitNone                 = 0x00,
	kThreadWaitKernelMutex          = 0x01,
	kThreadWaitPortReceive          = 0x02,
	kThreadWaitPortSetReceive       = 0x03,
	kThreadWaitPortSend             = 0x04,
	kThreadWaitPortSendInTransit    = 0x05,
	kThreadWaitSemaphore            = 0x06,
	kThreadWaitKernelRWLockRead     = 0x07,
	kThreadWaitKernelRWLockWrite    = 0x08,
	kThreadWaitKernelRWLockUpgrade  = 0x09,
	kThreadWaitUserLock             = 0x0a,
	kThreadWaitPThreadMutex         = 0x0b,
	kThreadWaitPThreadRWLockRead    = 0x0c,
	kThreadWaitPThreadRWLockWrite   = 0x0d,
	kThreadWaitPThreadCondVar       = 0x0e,
	kThreadWaitParkedWorkQueue      = 0x0f,
	kThreadWaitWorkloopSyncWait     = 0x10,
	kThreadWaitOnProcess            = 0x11,
	kThreadWaitSleepWithInheritor   = 0x12,
	kThreadWaitEventlink            = 0x13,
	kThreadWaitCompressor           = 0x14,
	kThreadWaitParkedBoundWorkQueue = 0x15,
	kThreadWaitPageBusy             = 0x16,
	kThreadWaitPagerInit            = 0x17,
	kThreadWaitPagerReady           = 0x18,
	kThreadWaitPagingActivity       = 0x19,
	kThreadWaitMappingInProgress    = 0x1a,
	kThreadWaitMemoryBlocked        = 0x1b,
	kThreadWaitPagingInProgress     = 0x1c,
	kThreadWaitPageInThrottle       = 0x1d,
	kThreadWaitExclaveCore          = 0x1e,
	kThreadWaitExclaveKit           = 0x1f,
} __attribute__((packed)) block_hint_t;

_Static_assert(sizeof(block_hint_t) <= sizeof(short),
    "block_hint_t must fit within a short");

#ifdef XNU_KERNEL_PRIVATE

struct turnstile;
struct waitq;
typedef struct stackshot_thread_waitinfo thread_waitinfo_t;
struct ipc_service_port_label;
struct portlabel_info;

/* Used for stackshot_thread_waitinfo_unsafe */
extern void kdp_lck_mtx_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_sema_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_mqueue_send_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_v2_t *waitinfo,
    struct ipc_service_port_label **isplp);
extern void kdp_mqueue_recv_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_v2_t *waitinfo,
    struct ipc_service_port_label **isplp);
extern void kdp_ulock_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_rwlck_find_owner(struct waitq * waitq, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_pthread_find_owner(thread_t thread, thread_waitinfo_t *waitinfo);
extern void *kdp_pthread_get_thread_kwq(thread_t thread);
extern void kdp_workloop_sync_wait_find_owner(thread_t thread, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_wait4_find_process(thread_t thread, event64_t event, thread_waitinfo_t *waitinfo);
extern void kdp_sleep_with_inheritor_find_owner(struct waitq * waitq, __unused event64_t event, thread_waitinfo_t * waitinfo);
extern void kdp_turnstile_fill_tsinfo(struct turnstile *ts, thread_turnstileinfo_v2_t *tsinfo, struct ipc_service_port_label **isplp);
extern void kdp_ipc_fill_splabel(struct ipc_service_port_label *ispl, struct portlabel_info *spl, const char **namep);
extern void kdp_ipc_splabel_size(size_t *ispl_size, size_t *maxnamelen);
extern const bool kdp_ipc_have_splabel;
void kdp_eventlink_find_owner(struct waitq *waitq, event64_t event, thread_waitinfo_t *waitinfo);
#if CONFIG_EXCLAVES
extern void kdp_esync_find_owner(struct waitq *waitq, event64_t event, thread_waitinfo_t *waitinfo);
#endif /* CONFIG_EXCLAVES */

#endif /* XNU_KERNEL_PRIVATE */

#endif /* !_KERN_BLOCK_HINT_H_ */
