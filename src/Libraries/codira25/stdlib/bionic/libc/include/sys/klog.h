/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
 * @file sys/klog.h
 * @brief
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/** Used with klogctl(). */
#define KLOG_CLOSE 0
/** Used with klogctl(). */
#define KLOG_OPEN 1
/** Used with klogctl(). */
#define KLOG_READ 2
/** Used with klogctl(). */
#define KLOG_READ_ALL 3
/** Used with klogctl(). */
#define KLOG_READ_CLEAR 4
/** Used with klogctl(). */
#define KLOG_CLEAR 5
/** Used with klogctl(). */
#define KLOG_CONSOLE_OFF 6
/** Used with klogctl(). */
#define KLOG_CONSOLE_ON 7
/** Used with klogctl(). */
#define KLOG_CONSOLE_LEVEL 8
/** Used with klogctl(). */
#define KLOG_SIZE_UNREAD 9
/** Used with klogctl(). */
#define KLOG_SIZE_BUFFER 10

/**
 * [klogctl(3)](https://man7.org/linux/man-pages/man2/klogctl.3.html) operates on the kernel log.
 *
 * This system call is not available to applications.
 * Use syslog() or `<android/log.h>` instead.
 */
int klogctl(int __type, char* __BIONIC_COMPLICATED_NULLNESS __buf, int __buf_size);

__END_DECLS
