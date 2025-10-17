/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#ifndef WebKit_WebKitAvailability_h
#define WebKit_WebKitAvailability_h

#import <TargetConditionals.h>

#if !TARGET_OS_IPHONE
#include <Foundation/NSObjCRuntime.h>

#define WEBKIT_AVAILABLE_MAC(introduced) NS_AVAILABLE_MAC(introduced)
#define WEBKIT_CLASS_AVAILABLE_MAC(introduced) NS_CLASS_AVAILABLE_MAC(introduced)
#define WEBKIT_ENUM_AVAILABLE_MAC(introduced) NS_ENUM_AVAILABLE_MAC(introduced)

#if !defined(DISABLE_LEGACY_WEBKIT_DEPRECATIONS) && !defined(BUILDING_WEBKIT)

#define WEBKIT_DEPRECATED_MAC(introduced, deprecated, ...) NS_DEPRECATED_MAC(introduced, deprecated, __VA_ARGS__)
#define WEBKIT_CLASS_DEPRECATED_MAC(introduced, deprecated, ...) NS_CLASS_DEPRECATED_MAC(introduced, deprecated, __VA_ARGS__)
#define WEBKIT_ENUM_DEPRECATED_MAC(introduced, deprecated, ...) NS_ENUM_DEPRECATED_MAC(introduced, deprecated, __VA_ARGS__)

#else

#define WEBKIT_DEPRECATED_MAC(introduced, deprecated, ...) NS_AVAILABLE_MAC(introduced)
#define WEBKIT_CLASS_DEPRECATED_MAC(introduced, deprecated, ...) NS_CLASS_AVAILABLE_MAC(introduced)
#define WEBKIT_ENUM_DEPRECATED_MAC(introduced, deprecated, ...) NS_ENUM_AVAILABLE_MAC(introduced)

#endif /* !defined(BUILDING_WEBKIT) && !defined(DISABLE_LEGACY_WEBKIT_DEPRECATIONS) */

#else

#define WEBKIT_AVAILABLE_MAC(introduced)
#define WEBKIT_CLASS_AVAILABLE_MAC(introduced)
#define WEBKIT_ENUM_AVAILABLE_MAC(introduced)
#define WEBKIT_DEPRECATED_MAC(introduced, deprecated, ...)
#define WEBKIT_CLASS_DEPRECATED_MAC(introduced, deprecated, ...)
#define WEBKIT_ENUM_DEPRECATED_MAC(introduced, deprecated, ...)

#endif /* !TARGET_OS_IPHONE */

#endif /* WebKit_WebKitAvailability_h */
