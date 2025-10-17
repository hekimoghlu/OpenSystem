/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
 * This file lists notification keys that are posted using the 
 * notify_post() API by various Mac OS X system services.
 * The exact circumstances under which services post these
 * notifications is controlled by those services, and may change
 * in future software releases.
 */

/*
 * Directory Service notifications
 * These are posted by the DirectoryService daemon to advise clients that
 * cached data should be invalidated.
 */
#define kNotifyDSCacheInvalidation "com.apple.system.DirectoryService.InvalidateCache"
#define kNotifyDSCacheInvalidationGroup "com.apple.system.DirectoryService.InvalidateCache.group"
#define kNotifyDSCacheInvalidationHost "com.apple.system.DirectoryService.InvalidateCache.host"
#define kNotifyDSCacheInvalidationService "com.apple.system.DirectoryService.InvalidateCache.service"
#define kNotifyDSCacheInvalidationUser "com.apple.system.DirectoryService.InvalidateCache.user"

/*
 * File System notifications
 * These advise clients of various filesystem events.
 */
#define kNotifyVFSMount "com.apple.system.kernel.mount"
#define kNotifyVFSUnmount "com.apple.system.kernel.unmount"
#define kNotifyVFSUpdate "com.apple.system.kernel.mountupdate"
#define kNotifyVFSLowDiskSpace "com.apple.system.lowdiskspace"
#define kNotifyVFSLowDiskSpaceRootFS "com.apple.system.lowdiskspace.system"
#define kNotifyVFSLowDiskSpaceOtherFS "com.apple.system.lowdiskspace.user"

/*
 * System Configuration notifications
 * These advise clients of changes in the system configuration
 * managed by the system configuration server (configd).
 * Note that a much richer set of notifications are available to
 * clients using the SCDynamicStore API.
 */
#define kNotifySCHostNameChange "com.apple.system.hostname"
#define kNotifySCNetworkChange "com.apple.system.config.network_change"

/*
 * ASL notifications
 * Sent by syslogd to advise clients that new log messages have been
 * added to the ASL database.
 */
#define kNotifyASLDBUpdate "com.apple.system.logger.message"

/*
 * Time Zone change notification
 * Sent by notifyd when the system's timezone changes.
 */
#define kNotifyTimeZoneChange "com.apple.system.timezone"

/*
 * System clock change notification
 * Sent when a process modifies the system clock using the settimeofday system call.
 */
#define kNotifyClockSet "com.apple.system.clock_set"
