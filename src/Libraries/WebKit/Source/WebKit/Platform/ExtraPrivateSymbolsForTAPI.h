/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#include <wtf/spi/darwin/XPCSPI.h>

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: Remove these after <rdar://problem/30772033> is fixed.
void NetworkServiceInitializer();
void WebContentServiceInitializer();
void GPUServiceInitializer();
void ModelServiceInitializer();

void ExtensionEventHandler(xpc_connection_t);

#if USE(EXTENSIONKIT)
// Declared in WKProcessExtension.h for use in extension targets. Must be declared in project
//  headers because the extension targets cannot import the entire WebKit module (rdar://119162443).
@interface WKGrant : NSObject
@end

@interface WKProcessExtension : NSObject
@end
#endif

#ifdef __cplusplus
}
#endif
