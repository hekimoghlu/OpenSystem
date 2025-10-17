/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#ifndef _UAPI_LINUX_PIDFD_H
#define _UAPI_LINUX_PIDFD_H
#include <linux/types.h>
#include <linux/fcntl.h>
#include <linux/ioctl.h>
#define PIDFD_NONBLOCK O_NONBLOCK
#define PIDFD_THREAD O_EXCL
#define PIDFD_SIGNAL_THREAD (1UL << 0)
#define PIDFD_SIGNAL_THREAD_GROUP (1UL << 1)
#define PIDFD_SIGNAL_PROCESS_GROUP (1UL << 2)
#define PIDFS_IOCTL_MAGIC 0xFF
#define PIDFD_GET_CGROUP_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 1)
#define PIDFD_GET_IPC_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 2)
#define PIDFD_GET_MNT_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 3)
#define PIDFD_GET_NET_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 4)
#define PIDFD_GET_PID_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 5)
#define PIDFD_GET_PID_FOR_CHILDREN_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 6)
#define PIDFD_GET_TIME_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 7)
#define PIDFD_GET_TIME_FOR_CHILDREN_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 8)
#define PIDFD_GET_USER_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 9)
#define PIDFD_GET_UTS_NAMESPACE _IO(PIDFS_IOCTL_MAGIC, 10)
#endif
