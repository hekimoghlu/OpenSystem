/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
 *
 * FILE: watchvol.h
 * AUTH: Soren Spies (sspies)
 * DATE: 6 March 2006
 * DESC: header for volume watching routines
 *
 */

// for kextd_main
int kextd_watch_volumes(void);
int kextd_giveup_volwatch();
void kextd_stop_volwatch();

void updateRAIDSet(
    CFNotificationCenterRef center,
    void * observer,
    CFStringRef name,
    const void * object,
    CFDictionaryRef userInfo);

void updateCoreStorageVolume(
    CFNotificationCenterRef center,
    void * observer,
    CFStringRef name,
    const void * object,
    CFDictionaryRef userInfo);
