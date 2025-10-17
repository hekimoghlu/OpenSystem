/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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

#if USE(LIBWEBRTC)

#if PLATFORM(IOS_FAMILY)
#define WEBRTC_IOS
#endif

#if PLATFORM(COCOA)
#define WEBRTC_MAC
#define ABSL_ALLOCATOR_NOTHROW 1
#endif

#define WEBRTC_WEBKIT_BUILD 1
#define WEBRTC_POSIX 1
#define _COMMON_INCLUDED_

#define WEBRTC_NON_STATIC_TRACE_EVENT_HANDLERS 0

#endif // USE(LIBWEBRTC)
