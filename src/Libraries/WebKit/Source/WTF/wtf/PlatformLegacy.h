/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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

#ifndef WTF_PLATFORM_GUARD_AGAINST_INDIRECT_INCLUSION
#error "Please #include <wtf/Platform.h> instead of this file directly."
#endif


/* PLATFORM() - handles OS, operating environment, graphics API, and
   CPU. This macro will be phased out in favor of platform adaptation
   macros, policy decision macros, and top-level port definitions. */
#define PLATFORM(WTF_FEATURE) (defined WTF_PLATFORM_##WTF_FEATURE && WTF_PLATFORM_##WTF_FEATURE)


/* FIXME: these are all mixes of OS, operating environment and policy choices. */
/* PLATFORM(GTK) */
/* PLATFORM(MAC) */
/* PLATFORM(IOS) */
/* PLATFORM(IOS_FAMILY) */
/* PLATFORM(IOS_SIMULATOR) */
/* PLATFORM(IOS_FAMILY_SIMULATOR) */
/* PLATFORM(WIN) */
#if defined(BUILDING_GTK__)
#define WTF_PLATFORM_GTK 1
#elif defined(BUILDING_HAIKU__)
#define WTF_PLATFORM_HAIKU 1
#elif defined(BUILDING_WPE__)
#define WTF_PLATFORM_WPE 1
#elif defined(BUILDING_JSCONLY__)
/* JSCOnly does not provide PLATFORM() macro */
#elif OS(MACOS)
#define WTF_PLATFORM_MAC 1
#elif OS(IOS_FAMILY)
#if OS(IOS)
/* PLATFORM(IOS) - iOS and iPadOS only (iPhone and iPad), not including macCatalyst, not including watchOS, not including tvOS, not including visionOS */
#define WTF_PLATFORM_IOS 1
#endif
/* PLATFORM(IOS_FAMILY) - iOS family, including iOS, iPadOS, macCatalyst, tvOS, watchOS, visionOS */
#define WTF_PLATFORM_IOS_FAMILY 1
#if TARGET_OS_SIMULATOR
#if OS(IOS)
#define WTF_PLATFORM_IOS_SIMULATOR 1
#endif
#define WTF_PLATFORM_IOS_FAMILY_SIMULATOR 1
#endif
#if defined(TARGET_OS_MACCATALYST) && TARGET_OS_MACCATALYST
#define WTF_PLATFORM_MACCATALYST 1
#endif
#elif OS(WINDOWS)
#define WTF_PLATFORM_WIN 1
#endif

/* PLATFORM(COCOA) */
#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)
#define WTF_PLATFORM_COCOA 1
#endif

/* PLATFORM(APPLETV) */
#if defined(TARGET_OS_TV) && TARGET_OS_TV
#define WTF_PLATFORM_APPLETV 1
#endif

/* PLATFORM(WATCHOS) */
#if defined(TARGET_OS_WATCH) && TARGET_OS_WATCH
#define WTF_PLATFORM_WATCHOS 1
#endif

/* PLATFORM(VISION) */
#if defined(TARGET_OS_VISION) && TARGET_OS_VISION
#define WTF_PLATFORM_VISION 1
#endif
