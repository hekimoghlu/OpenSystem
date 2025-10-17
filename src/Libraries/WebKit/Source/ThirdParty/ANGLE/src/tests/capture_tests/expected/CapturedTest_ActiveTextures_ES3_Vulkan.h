/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#include <EGL/egl.h>
#include <stdint.h>

// Public functions are declared in trace_fixture.h.

// Private Functions

void SetupReplayContext3(void);
void ReplayFrame1(void);
void ReplayFrame2(void);
void ReplayFrame3(void);
void ResetReplayContextShared(void);
void ResetReplayContext3(void);
void ReplayFrame4(void);
void SetupReplayContextShared(void);
void SetupReplayContextSharedInactive(void);
void InitReplay(void);

// Global variables

extern const char *const glShaderSource_string_0[];
extern const char *const glShaderSource_string_1[];
