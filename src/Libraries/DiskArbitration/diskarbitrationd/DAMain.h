/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef __DISKARBITRATIOND_DAMAIN__
#define __DISKARBITRATIOND_DAMAIN__

#include <sys/types.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern const char *           kDAMainMountPointFolder;
extern const char *           kDAMainMountPointFolderCookieFile;
extern const char *           kDAMainDataVolumeMountPointFolder;

extern CFURLRef               gDABundlePath;
extern CFStringRef            gDAConsoleUser;
extern gid_t                  gDAConsoleUserGID;
extern uid_t                  gDAConsoleUserUID;
extern CFArrayRef             gDAConsoleUserList;
extern CFMutableArrayRef      gDADiskList;
extern Boolean                gDAExit;
extern CFMutableArrayRef      gDAFileSystemList;
extern CFMutableArrayRef      gDAFileSystemProbeList;
extern Boolean                gDAIdle;
extern Boolean                gDAIdleTimerRunning;
extern CFAbsoluteTime         gDAIdleStartTime;
extern io_iterator_t          gDAMediaAppearedNotification;
extern io_iterator_t          gDAMediaDisappearedNotification;
extern IONotificationPortRef  gDAMediaPort;
extern CFMutableArrayRef      gDAMountMapList1;
extern CFMutableArrayRef      gDAMountMapList2;
extern CFMutableDictionaryRef gDAPreferenceList;
extern CFMutableArrayRef      gDAMountPointList;
extern CFMutableDictionaryRef gDADanglingVolumeList;
extern pid_t                  gDAProcessID;
extern char *                 gDAProcessName;
extern char *                 gDAProcessNameID;
extern CFMutableArrayRef      gDARequestList;
extern CFMutableArrayRef      gDAResponseList;
extern CFMutableArrayRef      gDASessionList;
extern CFMutableDictionaryRef gDAUnitList;
extern Boolean                gDAUnlockedState;

extern Boolean                gFSKitMissing;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAMAIN__ */
