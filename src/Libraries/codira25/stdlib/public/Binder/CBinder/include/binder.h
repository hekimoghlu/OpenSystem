/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#include <stdint.h>
#include <sys/types.h>

#ifdef __BIONIC__

#include <android/api-level.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/choreographer.h>
#include <android/native_activity.h>

#include <linux/android/binder.h>

// needed for AndroidSystem
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <sys/timerfd.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#endif

#ifdef __APPLE__

typedef uint8_t   __u8;
typedef uint16_t  __u16;
typedef uint32_t  __u32;
typedef uint64_t  __u64;

typedef int8_t    __s8;
typedef int16_t   __s16;
typedef int32_t   __s32;
typedef int64_t   __s64;

typedef pid_t     __kernel_pid_t;
typedef uid_t     __kernel_uid32_t;

#else

#include <linux/types.h>
#include <linux/ioctl.h>

#endif

// Kernel header, present in NDK but not found by default

#include "linux/android/binder.h"
#include "linux/android/binderfs.h"
