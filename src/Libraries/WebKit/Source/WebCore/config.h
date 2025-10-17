/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#if defined(HAVE_CONFIG_H) && HAVE_CONFIG_H && defined(BUILDING_WITH_CMAKE)
#include "cmakeconfig.h"
#endif

#include <wtf/Platform.h>

#if PLATFORM(COCOA)
#define USE_FILE_LOCK 1
#endif

#include "PlatformExportMacros.h"
#include <JavaScriptCore/JSExportMacros.h>
#include <pal/ExportMacros.h>

// Using CMake with Unix makefiles does not use prefix headers.
#if PLATFORM(MAC) && defined(BUILDING_WITH_CMAKE)
#include "WebCorePrefix.h"
#endif

#ifdef __cplusplus

// These undefs match up with defines in WebCorePrefix.h for Mac OS X.
// Helps us catch if anyone uses new or delete by accident in code and doesn't include "config.h".
#undef new
#undef delete
#include <wtf/FastMalloc.h>
#include <wtf/IsoMalloc.h>
#include <wtf/TZoneMalloc.h>

#include <ciso646>

#endif

#if USE(CG)
#ifndef CGFLOAT_DEFINED
#if (defined(__LP64__) && __LP64__) || (defined(__x86_64__) && __x86_64__) || defined(_M_X64) || defined(__amd64__)
typedef double CGFloat;
#else
typedef float CGFloat;
#endif
#define CGFLOAT_DEFINED 1
#endif
#endif /* USE(CG) */
