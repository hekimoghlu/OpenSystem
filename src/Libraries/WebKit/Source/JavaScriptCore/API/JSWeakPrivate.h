/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#ifndef JSWeakPrivate_h
#define JSWeakPrivate_h

#include <JavaScriptCore/JSObjectRef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef const struct OpaqueJSWeak* JSWeakRef;

JS_EXPORT JSWeakRef JSWeakCreate(JSContextGroupRef, JSObjectRef);

JS_EXPORT void JSWeakRetain(JSContextGroupRef, JSWeakRef);
JS_EXPORT void JSWeakRelease(JSContextGroupRef, JSWeakRef);

JS_EXPORT JSObjectRef JSWeakGetObject(JSWeakRef);

#ifdef __cplusplus
}
#endif

#endif // JSWeakPrivate_h

