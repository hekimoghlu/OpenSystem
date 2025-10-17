/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#ifndef WKTypes_h
#define WKTypes_h

#if TARGET_OS_IPHONE

#ifdef __cplusplus
extern "C" {
#endif

// This is named WAKObject to avoid a name conflict with WebKit's WKObject.
typedef struct _WAKObject WAKObject;
typedef struct _WAKObject *WAKObjectRef;
typedef struct WKControl* WKControlRef;
typedef struct _WKView* WKViewRef;

#ifdef __cplusplus
}
#endif

#endif // TARGET_OS_IPHONE

#endif // WKTypes_h
