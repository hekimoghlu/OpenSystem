/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#ifndef _SMBFS_NOTIFY_CHANGE_H_
#define _SMBFS_NOTIFY_CHANGE_H_

#define kNotifyThreadStarting	1
#define kNotifyThreadRunning	2
#define kNotifyThreadStopping	3
#define kNotifyThreadStop		4

enum  {
	kSendNotify = 1,
	kReceivedNotify = 2,
	kUsePollingToNotify = 3,
	kWaitingOnNotify = 4,
	kWaitingForRemoval = 5,
	kCancelNotify = 6
	
};

struct watch_item {
	lck_mtx_t		watch_statelock;
	uint32_t		state;
	void			*notify;
	vnode_t			vp;
	struct smb_ntrq *ntp;
	struct smb_rq   *rqp;           /* SMB 2/3 */
	uint64_t		flags;           /* Indicates if using SMB 2/3 or not */
	uint32_t		throttleBack;
	uint16_t		watchTree;
	int				isRoot;
    int             isServerMsg;
	struct timespec	last_notify_time;
	uint32_t		rcvd_notify_count;
	STAILQ_ENTRY(watch_item) entries;
};

struct smbfs_notify_change {
	struct smbmount		*smp;
	struct watch_item   *svrmsg_item;   /* SMB 2/3, for server messages */
	uint32_t			haveMoreWork;
	struct timespec		sleeptimespec;
	uint32_t			notify_state;
	int					pollOnly;		/* Server doesn't support notifications */
	int					watchCnt;		/* Count of all items on the list */
	int					watchPollCnt;	/* Count of all polling items on the list */
	lck_mtx_t			notify_statelock;
	lck_mtx_t			watch_list_lock;
	STAILQ_HEAD(, watch_item) watch_list;
};

#endif // _SMBFS_NOTIFY_CHANGE_H_
