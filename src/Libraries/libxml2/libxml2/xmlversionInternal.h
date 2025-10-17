/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#ifndef __XML_VERSION_INTERNAL_H__
#define __XML_VERSION_INTERNAL_H__

#include <stdbool.h>
#include <libxml/xmlversion.h>

extern bool linkedOnOrAfterFall2022OSVersions(void);
extern bool linkedOnOrAfter2024EReleases(void);

#ifdef __APPLE__
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 160000 \
    || defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 130000 \
    || defined(__TV_OS_VERSION_MIN_REQUIRED) && __TV_OS_VERSION_MIN_REQUIRED >= 160000 \
    || defined(__WATCH_OS_VERSION_MIN_REQUIRED) && __WATCH_OS_VERSION_MIN_REQUIRED >= 90000
#define LIBXML_HAS_XPOINTER_LOCATIONS_DISABLED_AT_RUNTIME
#define LIBXML_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16
#else
#undef LIBXML_HAS_XPOINTER_LOCATIONS_DISABLED_AT_RUNTIME
#undef LIBXML_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16
#endif

#if (defined(TARGET_OS_IOS) && TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (defined(TARGET_OS_MAC) && TARGET_OS_MAC && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150400) \
    || (defined(TARGET_OS_MACCATALYST) && TARGET_OS_MACCATALYST && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (defined(TARGET_OS_TV) && TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180400) \
    || (defined(TARGET_OS_VISION) && TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 20400) \
    || (defined(TARGET_OS_WATCH) && TARGET_OS_WATCH && __WATCH_OS_VERSION_MAX_ALLOWED >= 110400)
#define LIBXML_LINKED_ON_OR_AFTER_MACOS15_4_IOS18_4_WATCHOS11_4_TVOS18_4_VISIONOS2_4
#else
#undef LIBXML_LINKED_ON_OR_AFTER_MACOS15_4_IOS18_4_WATCHOS11_4_TVOS18_4_VISIONOS2_4
#endif

#else /* __APPLE__ */

#undef LIBXML_HAS_XPOINTER_LOCATIONS_DISABLED_AT_RUNTIME
#undef LIBXML_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16

#undef LIBXML_LINKED_ON_OR_AFTER_MACOS15_4_IOS18_4_WATCHOS11_4_TVOS18_4_VISIONOS2_4

#endif /* __APPLE__ */

#endif /* __XML_VERSION_INTERNAL_H__ */
