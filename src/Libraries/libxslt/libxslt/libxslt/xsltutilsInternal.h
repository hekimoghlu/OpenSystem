/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#ifndef __XML_XSLTUTILSINTERNAL_H__
#define __XML_XSLTUTILSINTERNAL_H__

#include <stdbool.h>
#include "xsltexports.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __clang_tapi__
bool linkedOnOrAfterFall2022OSVersions(void);
#endif

#ifdef __APPLE__
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 160000 \
    || defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 130000 \
    || defined(__TV_OS_VERSION_MIN_REQUIRED) && __TV_OS_VERSION_MIN_REQUIRED >= 160000 \
    || defined(__WATCH_OS_VERSION_MIN_REQUIRED) && __WATCH_OS_VERSION_MIN_REQUIRED >= 90000
#define LIBXSLT_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16
#else
#undef LIBXSLT_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16
#endif
#else /* __APPLE__ */
#undef LIBXSLT_LINKED_ON_OR_AFTER_MACOS13_IOS16_WATCHOS9_TVOS16
#endif /* __APPLE__ */

XSLTPUBFUN bool linkedOnOrAfterFall2023OSVersions(void);

#ifdef __APPLE__
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 170000 \
    || defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 140000 \
    || defined(__TV_OS_VERSION_MIN_REQUIRED) && __TV_OS_VERSION_MIN_REQUIRED >= 170000 \
    || defined(__WATCH_OS_VERSION_MIN_REQUIRED) && __WATCH_OS_VERSION_MIN_REQUIRED >= 100000
#define LIBXSLT_LINKED_ON_OR_AFTER_MACOS14_IOS17_WATCHOS10_TVOS17
#else
#undef LIBXSLT_LINKED_ON_OR_AFTER_MACOS14_IOS17_WATCHOS10_TVOS17
#endif
#else /* __APPLE__ */
#undef LIBXSLT_LINKED_ON_OR_AFTER_MACOS14_IOS17_WATCHOS10_TVOS17
#endif /* __APPLE__ */

/* Moved internal declarations from xsltutils.h. */

#ifdef LIBXSLT_API_FOR_MACOS14_IOS17_WATCHOS10_TVOS17

#define XSLT_SOURCE_NODE_MASK       15u
#define XSLT_SOURCE_NODE_HAS_KEY    1u
#define XSLT_SOURCE_NODE_HAS_ID     2u
#ifndef __clang_tapi__
int
xsltGetSourceNodeFlags(xmlNodePtr node);
int
xsltSetSourceNodeFlags(xsltTransformContextPtr ctxt, xmlNodePtr node,
                       int flags);
int
xsltClearSourceNodeFlags(xmlNodePtr node, int flags);
void **
xsltGetPSVIPtr(xmlNodePtr cur);
#endif /* __clang_tapi__ */

#endif /* LIBXSLT_API_FOR_MACOS14_IOS17_WATCHOS10_TVOS17 */

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLTUTILSINTERNAL_H__ */
