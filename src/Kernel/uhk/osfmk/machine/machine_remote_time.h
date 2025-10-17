/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#ifndef MACHINE_REMOTE_TIME_H
#define MACHINE_REMOTE_TIME_H

#if defined (__x86_64__)
#include "x86_64/machine_remote_time.h"
#elif defined (__arm64__)
#include "arm64/machine_remote_time.h"
#endif

#define BT_SLEEP_SENTINEL_TS  (~1ULL)
#define BT_WAKE_SENTINEL_TS   (~2ULL)
#define BT_RESET_SENTINEL_TS  (~3ULL)

#endif /* MACHINE_REMOTE_TIME_H */
