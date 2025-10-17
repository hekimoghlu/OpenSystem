/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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

#if ENABLE(WEBGL)

#include "IDLTypes.h"
#include "JSDOMConvertBase.h"

namespace WebCore {

JSC::JSValue convertToJSValue(JSC::JSGlobalObject&, JSDOMGlobalObject&, const WebGLAny&);
JSC::JSValue convertToJSValue(JSC::JSGlobalObject&, JSDOMGlobalObject&, WebGLExtensionAny);

template<> struct JSConverter<IDLWebGLAny> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const WebGLAny& value)
    {
        return convertToJSValue(lexicalGlobalObject, globalObject, value);
    }
};

template<> struct JSConverter<IDLWebGLExtensionAny> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, WebGLExtensionAny value)
    {
        return convertToJSValue(lexicalGlobalObject, globalObject, WTFMove(value));
    }
};

} // namespace WebCore

#endif
