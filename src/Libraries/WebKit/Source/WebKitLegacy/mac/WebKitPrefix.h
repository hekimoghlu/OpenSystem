/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include <TargetConditionals.h>

#ifdef __cplusplus
#define NULL __null
#else
#define NULL ((void *)0)
#endif

#import <stddef.h>
#import <stdio.h>
#import <fcntl.h>
#import <errno.h>
#import <unistd.h>
#import <signal.h>
#import <sys/types.h>
#import <sys/time.h>
#import <sys/resource.h>

#import <pthread.h>

#import <CoreGraphics/CoreGraphics.h>

#ifdef __cplusplus

#include <algorithm> // needed for exception_defines.h
#include <cstddef>
#include <new>

#endif

#if !TARGET_OS_IPHONE
#import <Carbon/Carbon.h>

#ifndef CGFLOAT_DEFINED
#ifdef __LP64__
typedef double CGFloat;
#else
typedef float CGFloat;
#endif
#define CGFLOAT_DEFINED 1
#endif
#endif

#ifdef __OBJC__
#if !TARGET_OS_IPHONE
#import <Cocoa/Cocoa.h>
#else
#import <Foundation/Foundation.h>
#endif
#endif

#include <JavaScriptCore/JSExportMacros.h>
#include <WebCore/PlatformExportMacros.h>
#include <pal/ExportMacros.h>
#include <wtf/ExportMacros.h>

/* Work around bug with C++ library that screws up Objective-C++ when exception support is disabled. */
#undef try
#undef catch

#ifdef __cplusplus
#include <wtf/FastMalloc.h>
#include <wtf/TZoneMalloc.h>
#endif
