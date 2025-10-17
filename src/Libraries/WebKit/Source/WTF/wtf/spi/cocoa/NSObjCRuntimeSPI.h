/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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

#if USE(APPLE_INTERNAL_SDK)
#include <Foundation/NSObjCRuntime_Private.h>
#endif

#include <wtf/Platform.h>

// Apply this to a specific method in the @interface or @implementation.
#ifndef NS_DIRECT
#if HAVE(NS_DIRECT_SUPPORT)
#define NS_DIRECT __attribute__((objc_direct))
#else
#define NS_DIRECT
#endif
#endif

// Apply this to the @interface or @implementation of a class.
#ifndef NS_DIRECT_MEMBERS
#if HAVE(NS_DIRECT_SUPPORT)
#define NS_DIRECT_MEMBERS __attribute__((objc_direct_members))
#else
#define NS_DIRECT_MEMBERS
#endif
#endif
