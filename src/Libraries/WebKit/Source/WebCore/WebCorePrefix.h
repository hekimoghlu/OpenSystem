/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
/* This prefix file should contain only:
 *    1) files to precompile for faster builds
 *    2) in one case at least: OS-X-specific performance bug workarounds
 *    3) the special trick to catch us using new or delete without including "config.h"
 * The project should be able to build without this header, although we rarely test that.
 */

/* Things that need to be defined globally should go into "config.h". */

#if defined(HAVE_CONFIG_H) && HAVE_CONFIG_H && defined(BUILDING_WITH_CMAKE)
#include "cmakeconfig.h"
#endif

#include <wtf/Platform.h>

#if defined(__APPLE__)
#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif
#endif

#if !OS(WINDOWS)
#include <pthread.h>
#endif // !OS(WINDOWS)

#include <sys/types.h>
#include <fcntl.h>
#if HAVE(REGEX_H)
#include <regex.h>
#endif

#include <setjmp.h>

#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <unistd.h>
#endif

#ifdef __cplusplus
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <typeinfo>
#endif

#if defined(__APPLE__)
#include <sys/param.h>
#endif
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/time.h>
#include <sys/resource.h>
#endif

#if USE(CF)
#include <CoreFoundation/CoreFoundation.h>
#endif

#if USE(CG)
#include <CoreGraphics/CoreGraphics.h>
#endif

#if OS(WINDOWS)
#ifndef CF_IMPLICIT_BRIDGING_ENABLED
#define CF_IMPLICIT_BRIDGING_ENABLED
#endif

#ifndef CF_IMPLICIT_BRIDGING_DISABLED
#define CF_IMPLICIT_BRIDGING_DISABLED
#endif

#if USE(CF)
#include <CoreFoundation/CFBase.h>
#endif

#ifndef CF_ENUM
#define CF_ENUM(_type, _name) _type _name; enum
#endif
#ifndef CF_OPTIONS
#define CF_OPTIONS(_type, _name) _type _name; enum
#endif
#ifndef CF_ENUM_DEPRECATED
#define CF_ENUM_DEPRECATED(_macIntro, _macDep, _iosIntro, _iosDep)
#endif
#ifndef CF_ENUM_AVAILABLE
#define CF_ENUM_AVAILABLE(_mac, _ios)
#endif
#endif

#if PLATFORM(WIN)
#include <windows.h>
#else

#if OS(WINDOWS)
#include <windows.h>
#endif // OS(WINDOWS)

#if USE(OS_LOG)
#include <os/log.h>
#endif

#if PLATFORM(IOS_FAMILY)
#include <MobileCoreServices/MobileCoreServices.h>
#endif

#if PLATFORM(MAC)
#if !USE(APPLE_INTERNAL_SDK)
/* SecTrustedApplication.h declares SecTrustedApplicationCreateFromPath(...) to
 * be unavailable on macOS, so do not include that header. */
#define _SECURITY_SECTRUSTEDAPPLICATION_H_
#endif
#include <CoreServices/CoreServices.h>
#endif

#endif

#ifdef __OBJC__
#if PLATFORM(IOS_FAMILY)
#import <Foundation/Foundation.h>
#else
#if USE(APPKIT)
#import <Cocoa/Cocoa.h>
#endif
#endif // PLATFORM(IOS_FAMILY)
#endif

#ifdef __cplusplus

#if !PLATFORM(WIN)
#include <wtf/FastMalloc.h>
#include <wtf/HashMap.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/WTFString.h>
#endif

#define new ("if you use new/delete make sure to include config.h at the top of the file"()) 
#define delete ("if you use new/delete make sure to include config.h at the top of the file"()) 
#endif

/* When C++ exceptions are disabled, the C++ library defines |try| and |catch|
 * to allow C++ code that expects exceptions to build. These definitions
 * interfere with Objective-C++ uses of Objective-C exception handlers, which
 * use |@try| and |@catch|. As a workaround, undefine these macros. */
#ifdef __OBJC__
#undef try
#undef catch
#endif
