/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#if !defined(__WEBKITDOM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/webkitdom.h> can be included directly."
#endif

#ifndef WebKitDOMDefines_h
#define WebKitDOMDefines_h

#include <glib.h>

#ifdef G_OS_WIN32
    #ifdef BUILDING_WEBKIT
        #define WEBKIT_API __declspec(dllexport)
    #else
        #define WEBKIT_API __declspec(dllimport)
    #endif
#else
    #define WEBKIT_API __attribute__((visibility("default")))
#endif

#define WEBKIT_DEPRECATED WEBKIT_API G_DEPRECATED
#define WEBKIT_DEPRECATED_FOR(f) WEBKIT_API G_DEPRECATED_FOR(f)

#ifndef WEBKIT_API
    #define WEBKIT_API
#endif

#endif /* WebKitDOMDefines_h */
