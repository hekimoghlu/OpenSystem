/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#ifndef __WEBDAV_UTILS__
#define __WEBDAV_UTILS__

#include "webdav.h"

enum webdavlocktype  {WEBDAV_SHARED_LOCK = 1, WEBDAV_EXCLUSIVE_LOCK = 2};

/* single */
int webdav_lock(struct webdavnode *pt, enum webdavlocktype locktype);
void webdav_unlock(struct webdavnode *pt);

// convert standard timespec to webdav_timespec_64
void timespec_to_webdav_timespec64(struct timespec ts, struct webdav_timespec64 *wts);
// convert webdav_timespec64 to standard timespec
void webdav_timespec64_to_timespec(struct webdav_timespec64 wts, struct timespec *ts);

#endif