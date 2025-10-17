/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// trace.h: Wrappers for ANGLE trace event functions.
//

#ifndef LIBANGLE_TRACE_H_
#define LIBANGLE_TRACE_H_

#include <platform/PlatformMethods.h>
#include "common/base/anglebase/trace_event/trace_event.h"

// TODO: Pass platform directly to these methods. http://anglebug.com/42260698
#define ANGLE_TRACE_EVENT_BEGIN(CATEGORY, EVENT, ...) \
    TRACE_EVENT_BEGIN(ANGLEPlatformCurrent(), CATEGORY, EVENT, ##__VA_ARGS__)

#define ANGLE_TRACE_EVENT_END(CATEGORY, EVENT, ...) \
    TRACE_EVENT_END(ANGLEPlatformCurrent(), CATEGORY, EVENT, ##__VA_ARGS__)

#define ANGLE_TRACE_EVENT_INSTANT(CATEGORY, EVENT, ...) \
    TRACE_EVENT_INSTANT(ANGLEPlatformCurrent(), CATEGORY, EVENT, ##__VA_ARGS__)

#define ANGLE_TRACE_EVENT(CATEGORY, EVENT, ...) \
    TRACE_EVENT(ANGLEPlatformCurrent(), CATEGORY, EVENT, ##__VA_ARGS__)

// Deprecated, use ANGLE_TRACE_EVENT_BEGIN
#define ANGLE_TRACE_EVENT_BEGIN0(CATEGORY, EVENT) ANGLE_TRACE_EVENT_BEGIN(CATEGORY, EVENT)
// Deprecated, use ANGLE_TRACE_EVENT_END
#define ANGLE_TRACE_EVENT_END0(CATEGORY, EVENT) ANGLE_TRACE_EVENT_END(CATEGORY, EVENT)
// Deprecated, use ANGLE_TRACE_EVENT_INSTANT
#define ANGLE_TRACE_EVENT_INSTANT0(CATEGORY, EVENT) ANGLE_TRACE_EVENT_INSTANT(CATEGORY, EVENT)
// Deprecated, use ANGLE_TRACE_EVENT
#define ANGLE_TRACE_EVENT0(CATEGORY, EVENT) ANGLE_TRACE_EVENT(CATEGORY, EVENT)
// Deprecated, use ANGLE_TRACE_EVENT
#define ANGLE_TRACE_EVENT1(CATEGORY, EVENT, NAME, VAL) ANGLE_TRACE_EVENT(CATEGORY, EVENT, NAME, VAL)

#endif  // LIBANGLE_TRACE_H_
