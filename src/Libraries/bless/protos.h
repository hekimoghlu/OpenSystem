/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
 *  protos.h
 *  bless
 *
 *  Created by Shantonu Sen on 6/6/06.
 *  Copyright 2006-2007 Apple Inc. All Rights Reserved.
 *
 */

#include "enums.h"
#include "structs.h"
#include <stdbool.h>
#include <sys/cdefs.h>

#define BSD_NAME_SIZE 128

BLPreBootEnvType getPrebootType(void);
int modeInfo(BLContextPtr context, struct clarg actargs[klast]);
int modeDevice(BLContextPtr context, struct clarg actargs[klast]);
int modeFolder(BLContextPtr context, struct clarg actargs[klast]);
int modeFirmware(BLContextPtr context, struct clarg actargs[klast]);
int modeNetboot(BLContextPtr context, struct clarg actargs[klast]);
int modeUnbless(BLContextPtr context, struct clarg actargs[klast]);
int extractMountPoint(BLContextPtr context, struct clarg actargs[klast]);
int extractDiskFromMountPoint(BLContextPtr context, const char *mnt, char *disk, size_t disk_size);
int isMediaExternal(BLContextPtr context, const char *mnt, bool *external);
int isMediaRemovable(BLContextPtr context, const char *mnt, bool *removable);
int isMediaTDM(BLContextPtr context, const char *mnt, bool *tdm);

int blessViaBootability(BLContextPtr context, struct clarg actargs[klast]);

int blesslog(void *context, int loglevel, const char *string);
int blesscontextprintf(BLContextPtr context, int loglevel, char const *fmt, ...) __printflike(3, 4);

void usage(void);
void usage_short(void);

void addPayload(const char *path);

int CopyManifests(BLContextPtr context, const char *destPath, const char *srcPath, const char *srcSystemPath);
int PersonalizeOSVolume(BLContextPtr context, const char *volumePath, const char *prFile, bool suppressACPrompt);


extern int setboot(BLContextPtr context, char *device, CFDataRef bootxData,
				   CFDataRef labelData);
extern int setefilegacypath(BLContextPtr context, const char * path, int bootNext,
                            const char *legacyHint, const char *optionalData);

int BlessPrebootVolume(BLContextPtr context, const char *rootBSD, const char *bootEFISourceLocation,
					   CFDataRef labelData, CFDataRef labelData2, bool supportLegacy, struct clarg actargs[klast]);
int GetVolumeUUIDs(BLContextPtr context, const char *volBSD, CFStringRef *volUUID, CFStringRef *groupUUID);
int GetMountForSnapshot(BLContextPtr context, const char *snapshotName, const char *bsd, char *mountPoint, int mountPointLen);
int WriteLabelFile(BLContextPtr context, const char *path, CFDataRef labeldata, int doTypeCreator, int scale);
int GetSnapshotNameFromRootHash(BLContextPtr context, const char *rootHashPath, char *snapName, int nameLen);

int DeleteFileOrDirectory(const char *path);
