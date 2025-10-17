/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Portions Copyright 2007-2011 Apple Inc.
 */

/*
 * Information kept for each trigger vnode.
 * This is opaque outside the autofs project.
 */
struct trigger_info {
	TAILQ_ENTRY(trigger_info) ti_entries;
	/* tail queue of resolved triggers */
	lck_mtx_t       *ti_lock;       /* mutex protecting accesses */
	uint32_t        ti_seq;         /* sequence number of state changes */
	u_int           ti_flags;
	int             ti_error;
	fsid_t          ti_this_fsid;   /* fsid of file system for this trigger */
	fsid_t          ti_mounted_fsid;
	/* fsid of file system mounted atop this */
	time_t          ti_ref_time;    /* last reference to this trigger */
	int             (*ti_check_notrigger_process)(int);
	/* call this to check whether this process should trigger mounts */
	int             (*ti_check_nowait_process)(int);
	/* call this to check whether this process should block on mount-in-progress */
	int             (*ti_check_homedirmounter_process)(vnode_t, int);
	/* call this to check whether this process is a home directory mounter */
	int             (*ti_check_homedirmount)(vnode_t);
	/* call this to check whether this trigger is having a home directory mount done */
	void            *(*ti_get_mount_args)(vnode_t, vfs_context_t, int *);
	/* call this to get mount arguments */
	int             (*ti_do_mount)(void *);
	/* call this to make the mount upcall */
	void            (*ti_rel_mount_args)(void *);
	/* call this to release mount arguments */
	void            (*ti_rearm)(vnode_t, int);
	/* call this on a rearm */
	void            (*ti_reclaim)(void *);
	/* call this on a reclaim */
	void            *ti_private;    /* private data, if any */
};

#define TF_INPROG               0x00000001      /* a mount is in progress for this trigger */
#define TF_WAITING              0x00000002      /* somebody's waiting for that mount to finish */
#define TF_FORCEMOUNT           0x00000004      /* all operations cause a mount */
#define TF_AUTOFS               0x00000008      /* an autofs mount will be done atop this trigger */
#define TF_DONTUNMOUNT          0x00000010      /* don't auto-unmount or preemptively unmount this */
#define TF_DONTPREUNMOUNT       0x00000020      /* don't preemptively unmount this */
#define TF_RESOLVED             0x00000040      /* trigger is on the resolved list */

/*
 * Call used by the automounter to specify some additional routines
 * to call.
 */
extern trigger_info_t *trigger_new_autofs(struct vnode_trigger_param *vnt,
    u_int flags,
    int (*check_notrigger_process)(int),
    int (*check_nowait_process)(int),
    int (*check_homedirmounter_process)(vnode_t, int),
    int (*check_homedirmount)(vnode_t),
    void *(*get_mount_args)(vnode_t, vfs_context_t, int *),
    int (*do_mount)(void *),
    void (*rel_mount_args)(void *),
    void (*rearm)(vnode_t, int),
    void (*reclaim)(void *),
    void *private);

/*
 * Set the mount timeout.
 */
extern void trigger_set_mount_to(int);

extern int auto_get_automountd_port(mach_port_t *automount_port);
extern void auto_release_port(mach_port_t port);
extern kern_return_t auto_new_thread(void (*)(void *), void *);

/*
 * Look at all triggered mounts, and, for each mount, if it is an
 * unconditional operation or if it hasn't been referred to recently,
 * unmount it.
 */
extern void unmount_triggered_mounts(int unconditional);
