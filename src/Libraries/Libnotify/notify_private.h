/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#ifndef __NOTIFY_PRIVATE_H__
#define __NOTIFY_PRIVATE_H__

#include <stdint.h>
#include <sys/types.h>
#include <os/base.h>
#include <Availability.h>

#define NOTIFY_OPT_DISPATCH 0x00000001
#define NOTIFY_OPT_REGEN    0x00000002
#define NOTIFY_OPT_FILTERED 0x00000004
#define NOTIFY_OPT_OLD_IPC  0x00000008
#define NOTIFY_OPT_ENABLE   0x04000000
#define NOTIFY_OPT_DISABLE  0x08000000

#define NOTIFY_NO_DISPATCH  0x80000000

#define ROOT_ENTITLEMENT_KEY "com.apple.notify.root_access"

OS_EXPORT uint32_t notify_suspend_pid(pid_t pid)
__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_4_0);

OS_EXPORT uint32_t notify_resume_pid(pid_t pid)
__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_4_0);

OS_EXPORT uint32_t notify_simple_post(const char *name)
__API_DEPRECATED("No longer supported, use notify_post", macos(10.7, 10.15), ios(4.3, 13.0), watchos(1.0, 6.0), tvos(1.0, 13.0));

OS_EXPORT void notify_set_options(uint32_t opts)
__OSX_AVAILABLE_STARTING(__MAC_10_8,__IPHONE_6_0);

OS_EXPORT void _notify_fork_child(void)
__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_4_3);

OS_EXPORT uint32_t notify_peek(int token, uint32_t *val)
__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_4_3);

// This SPI requires a sandbox exception in notifyd and will fail silently for
// new clients or new filepaths from existing clients. It is reccomended that
// both existing and new clients use some other file monitoring system, such as
// dispatch_source or FSEvents.
OS_EXPORT uint32_t notify_monitor_file(int token, char *path, int flags)
__API_DEPRECATED("No longer supported for new clients", macos(10.7, 10.16), ios(4.3, 14.0), watchos(1.0, 7.0), tvos(1.0, 14.0));

OS_EXPORT uint32_t notify_get_event(int token, int *ev, char *buf, int *len)
__API_DEPRECATED("No longer supported", macos(10.7, 10.15), ios(4.3, 13.0), watchos(1.0, 6.0), tvos(1.0, 13.0));

OS_EXPORT uint32_t notify_register_plain(const char *name, int *out_token)
__OSX_AVAILABLE_STARTING(__MAC_10_7,__IPHONE_4_3);

OS_EXPORT uint32_t notify_dump_status(const char *filepath);

#endif /* __NOTIFY_PRIVATE_H__ */
