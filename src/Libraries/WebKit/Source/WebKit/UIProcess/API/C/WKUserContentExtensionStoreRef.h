/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#ifndef WKUserContentExtensionStoreRef_h
#define WKUserContentExtensionStoreRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKUserContentExtensionStoreGetTypeID();

WK_EXPORT WKUserContentExtensionStoreRef WKUserContentExtensionStoreCreate(WKStringRef path);

typedef uint32_t WKUserContentExtensionStoreResult;
enum {
    kWKUserContentExtensionStoreSuccess = 0,
    kWKUserContentExtensionStoreLookupFailed,
    kWKUserContentExtensionStoreVersionMismatch,
    kWKUserContentExtensionStoreCompileFailed,
    kWKUserContentExtensionStoreRemoveFailed,
};

typedef void (*WKUserContentExtensionStoreFunction)(WKUserContentFilterRef, WKUserContentExtensionStoreResult, void*);
WK_EXPORT void WKUserContentExtensionStoreCompile(WKUserContentExtensionStoreRef, WKStringRef identifier, WKStringRef jsonSource, void* context, WKUserContentExtensionStoreFunction callback);
WK_EXPORT void WKUserContentExtensionStoreLookup(WKUserContentExtensionStoreRef, WKStringRef identifier, void* context, WKUserContentExtensionStoreFunction callback);
WK_EXPORT void WKUserContentExtensionStoreRemove(WKUserContentExtensionStoreRef, WKStringRef identifier, void* context, WKUserContentExtensionStoreFunction callback);

#ifdef __cplusplus
}
#endif

#endif /* WKUserContentExtensionStoreRef_h */
