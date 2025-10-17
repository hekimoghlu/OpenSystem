/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#if !PLATFORM(WIN)
#error "This file should only be included when building the Windows port."
#endif


/* --------- Windows port --------- */


#if !defined(ENABLE_GEOLOCATION)
#define ENABLE_GEOLOCATION 1
#endif

#if !defined(ENABLE_OPENTYPE_MATH)
#define ENABLE_OPENTYPE_MATH 1
#endif

#if !defined(ENABLE_WEB_ARCHIVE)
#define ENABLE_WEB_ARCHIVE 1
#endif

#if !defined(ENABLE_WEBGL)
#define ENABLE_WEBGL 1
#endif

#if !defined(ENABLE_WEBPROCESS_CACHE)
#define ENABLE_WEBPROCESS_CACHE 1
#endif
