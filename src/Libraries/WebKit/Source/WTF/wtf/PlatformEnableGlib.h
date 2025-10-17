/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

#if !PLATFORM(GTK) && !PLATFORM(WPE)
#error "This file should only be included when building for the GTK or WPE platforms."
#endif

/* Please keep the following in alphabetical order so we can notice duplicates. */
/* Items should only be here if they are different from the defaults in PlatformEnable.h. */

#if !defined(ENABLE_KINETIC_SCROLLING) && (ENABLE(ASYNC_SCROLLING) || PLATFORM(GTK))
#define ENABLE_KINETIC_SCROLLING 1
#endif

#if !defined(ENABLE_NOTIFICATION_EVENT) && ENABLE(NOTIFICATIONS)
#define ENABLE_NOTIFICATION_EVENT 1
#endif

#if !defined(ENABLE_OPENTYPE_VERTICAL)
#define ENABLE_OPENTYPE_VERTICAL 1
#endif

#if !defined(ENABLE_SCROLLING_THREAD) && USE(COORDINATED_GRAPHICS)
#define ENABLE_SCROLLING_THREAD 1
#endif

#if !defined(ENABLE_PDFJS)
#define ENABLE_PDFJS 1
#endif

#if !defined(ENABLE_WEBPROCESS_CACHE)
#define ENABLE_WEBPROCESS_CACHE 1
#endif

#if ENABLE(WPE_PLATFORM) || PLATFORM(GTK)
#define ENABLE_DAMAGE_TRACKING 1
#endif
