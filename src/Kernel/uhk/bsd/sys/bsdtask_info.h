/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#ifndef _SYS_BSDTASK_INFO_H
#define _SYS_BSDTASK_INFO_H

#include <vm/vm_map.h>

struct proc_taskinfo_internal {
	uint64_t                pti_virtual_size;   /* virtual memory size (bytes) */
	uint64_t                pti_resident_size;  /* resident memory size (bytes) */
	uint64_t                pti_total_user;         /* total time */
	uint64_t                pti_total_system;
	uint64_t                pti_threads_user;       /* existing threads only */
	uint64_t                pti_threads_system;
	int32_t                 pti_policy;             /* default policy for new threads */
	int32_t                 pti_faults;             /* number of page faults */
	int32_t                 pti_pageins;    /* number of actual pageins */
	int32_t                 pti_cow_faults; /* number of copy-on-write faults */
	int32_t                 pti_messages_sent;      /* number of messages sent */
	int32_t                 pti_messages_received; /* number of messages received */
	int32_t                 pti_syscalls_mach;  /* number of mach system calls */
	int32_t                 pti_syscalls_unix;  /* number of unix system calls */
	int32_t                 pti_csw;            /* number of context switches */
	int32_t                 pti_threadnum;          /* number of threads in the task */
	int32_t                 pti_numrunning;         /* number of running threads */
	int32_t                 pti_priority;           /* task priority*/
};

#define MAXTHREADNAMESIZE 64

struct proc_threadinfo_internal {
	uint64_t                pth_user_time;      /* user run time */
	uint64_t                pth_system_time;    /* system run time */
	int32_t                 pth_cpu_usage;      /* scaled cpu usage percentage */
	int32_t                 pth_policy;             /* scheduling policy in effect */
	int32_t                 pth_run_state;      /* run state (see below) */
	int32_t                 pth_flags;          /* various flags (see below) */
	int32_t                 pth_sleep_time;     /* number of seconds that thread */
	int32_t                 pth_curpri;             /* cur priority*/
	int32_t                 pth_priority;           /*  priority*/
	int32_t                 pth_maxpriority;                /* max priority*/
	char                    pth_name[MAXTHREADNAMESIZE];            /* thread name, if any */
};

struct proc_threadschedinfo_internal {
	uint64_t               int_time_ns;         /* time spent in interrupt context */
};


struct proc_regioninfo_internal {
	uint32_t                pri_protection;
	uint32_t                pri_max_protection;
	uint32_t                pri_inheritance;
	uint32_t                pri_flags;              /* shared, external pager, is submap */
	uint64_t                pri_offset;
	uint32_t                pri_behavior;
	uint32_t                pri_user_wired_count;
	uint32_t                pri_user_tag;
	uint32_t                pri_pages_resident;
	uint32_t                pri_pages_shared_now_private;
	uint32_t                pri_pages_swapped_out;
	uint32_t                pri_pages_dirtied;
	uint32_t                pri_ref_count;
	uint32_t                pri_shadow_depth;
	uint32_t                pri_share_mode;
	uint32_t                pri_private_pages_resident;
	uint32_t                pri_shared_pages_resident;
	uint32_t                pri_obj_id;
	uint32_t                pri_depth;
	uint64_t                pri_address;
	uint64_t                pri_size;
};

#ifdef  MACH_KERNEL_PRIVATE

#define PROC_REGION_SUBMAP      1
#define PROC_REGION_SHARED      2

extern uint32_t vnode_vid(void *vp);

#if CONFIG_IOSCHED
extern struct vnode *vnode_mountdevvp(struct vnode *);
#endif

extern boolean_t vnode_isonexternalstorage(void *vp);

#endif /* MACH_KERNEL_PRIVATE */

extern int fill_procregioninfo(task_t t, uint64_t arg, struct proc_regioninfo_internal *pinfo, uintptr_t *vp, uint32_t *vid);
extern int fill_procregioninfo_onlymappedvnodes(task_t t, uint64_t arg, struct proc_regioninfo_internal *pinfo, uintptr_t *vp, uint32_t *vid);
void fill_taskprocinfo(task_t task, struct proc_taskinfo_internal * ptinfo);
int fill_taskthreadinfo(task_t task, uint64_t thaddr, bool thuniqueid, struct proc_threadinfo_internal * ptinfo, void *, int *);
int fill_taskthreadlist(task_t task, void * buffer, int thcount, bool thuniqueid);
int fill_taskthreadschedinfo(task_t task, uint64_t thaddr, struct proc_threadschedinfo_internal *thread_sched_info);
int get_numthreads(task_t);
boolean_t bsd_hasthreadname(void *uth);
void bsd_getthreadname(void *uth, char* buffer);
void bsd_setthreadname(void *uth, uint64_t tid, const char* buffer);
void bsd_threadcdir(void * uth, void *vptr, int *vidp);
extern void bsd_copythreadname(void *dst_uth, void *src_uth);
int fill_taskipctableinfo(task_t task, uint32_t *table_size, uint32_t *table_free);

#endif /*_SYS_BSDTASK_INFO_H */
