/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMMON_EVENT_TRACER_H_
#define COMMON_EVENT_TRACER_H_

#include "common/platform.h"
#include "platform/PlatformMethods.h"

namespace angle
{
const unsigned char *GetTraceCategoryEnabledFlag(PlatformMethods *platform, const char *name);
angle::TraceEventHandle AddTraceEvent(PlatformMethods *platform,
                                      char phase,
                                      const unsigned char *categoryGroupEnabled,
                                      const char *name,
                                      unsigned long long id,
                                      int numArgs,
                                      const char **argNames,
                                      const unsigned char *argTypes,
                                      const unsigned long long *argValues,
                                      unsigned char flags);
}  // namespace angle

#endif  // COMMON_EVENT_TRACER_H_
