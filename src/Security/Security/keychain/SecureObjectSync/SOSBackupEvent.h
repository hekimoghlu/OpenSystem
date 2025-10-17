/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
//
//  SOSBackupEvent.h
//  sec
//

#ifndef _sec_SOSBackupEvent_
#define _sec_SOSBackupEvent_

#include <stdbool.h>
#include <stdio.h>
#include <CoreFoundation/CoreFoundation.h>

bool SOSBackupEventWriteReset(FILE *journalFile, CFDataRef keybag, CFErrorRef *error);
bool SOSBackupEventWriteDelete(FILE *journalFile, CFDataRef deletedDigest, CFErrorRef *error);
bool SOSBackupEventWriteAdd(FILE *journalFile, CFDictionaryRef backup_item, CFErrorRef *error);
bool SOSBackupEventWriteCompleteMarker(FILE *journalFile, uint64_t eventID, CFErrorRef *error);

#endif /* defined(_sec_SOSBackupEvent_) */
