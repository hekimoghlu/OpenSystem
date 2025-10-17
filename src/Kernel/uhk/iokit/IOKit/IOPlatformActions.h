/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

#include <IOKit/IOReturn.h>

extern "C" kern_return_t IOCPURunPlatformQuiesceActions(void);
extern "C" kern_return_t IOCPURunPlatformActiveActions(void);
extern "C" kern_return_t IOCPURunPlatformHaltRestartActions(uint32_t message);
extern "C" kern_return_t IOCPURunPlatformPanicActions(uint32_t message, uint32_t details);
extern "C" kern_return_t IOCPURunPlatformPanicSyncAction(void *addr, uint32_t offset, uint32_t len);

void IOPlatformActionsPreSleep(void);
void IOPlatformActionsPostResume(void);
