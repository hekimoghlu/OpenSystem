/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#pragma once

/**
 * @file sys/msg.h
 * @brief System V message queues. Not useful on Android because it's disallowed by SELinux.
 */

#include <sys/cdefs.h>
#include <sys/ipc.h>

#include <linux/msg.h>

#define msqid_ds msqid64_ds

__BEGIN_DECLS

typedef __kernel_ulong_t msgqnum_t;
typedef __kernel_ulong_t msglen_t;

/** Not useful on Android; disallowed by SELinux. */

#if __BIONIC_AVAILABILITY_GUARD(26)
int msgctl(int __msg_id, int __op, struct msqid_ds* _Nullable __buf) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
int msgget(key_t __key, int __flags) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
ssize_t msgrcv(int __msg_id, void* _Nonnull __msgbuf_ptr, size_t __size, long __type, int __flags) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
int msgsnd(int __msg_id, const void* _Nonnull __msgbuf_ptr, size_t __size, int __flags) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS
