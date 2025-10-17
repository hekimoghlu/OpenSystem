/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include <wtf/Vector.h>

#define CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING 0

#define CONTENT_EXTENSIONS_PERFORMANCE_REPORTING 0

#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
typedef CrashOnOverflow ContentExtensionsOverflowHandler;
#else
typedef UnsafeVectorOverflow ContentExtensionsOverflowHandler;
#endif

#if CONTENT_EXTENSIONS_PERFORMANCE_REPORTING
#define LOG_LARGE_STRUCTURES(name, size) if (size > 1000000) { dataLogF("NAME: %s SIZE %d\n", #name, (int)(size)); };
#else
#define LOG_LARGE_STRUCTURES(name, size)
#endif

#endif // ENABLE(CONTENT_EXTENSIONS)
