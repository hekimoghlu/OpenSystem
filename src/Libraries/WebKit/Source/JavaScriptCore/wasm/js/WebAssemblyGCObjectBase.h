/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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

#if ENABLE(WEBASSEMBLY)

#include "JSObject.h"
#include "WasmTypeDefinition.h"

namespace JSC {

class WebAssemblyGCObjectBase : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesGetPrototype | OverridesPut | OverridesIsExtensible | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;

    DECLARE_EXPORT_INFO;

    RefPtr<const Wasm::RTT> rtt() { return m_rtt; }

    static constexpr ptrdiff_t offsetOfRTT() { return OBJECT_OFFSETOF(WebAssemblyGCObjectBase, m_rtt); };

protected:
    DECLARE_VISIT_CHILDREN;

    WebAssemblyGCObjectBase(VM&, Structure*, RefPtr<const Wasm::RTT>&&);

    DECLARE_DEFAULT_FINISH_CREATION;

    JS_EXPORT_PRIVATE static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    JS_EXPORT_PRIVATE static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned, PropertySlot&);
    JS_EXPORT_PRIVATE static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    JS_EXPORT_PRIVATE static bool putByIndex(JSCell*, JSGlobalObject*, unsigned, JSValue, bool shouldThrow);
    JS_EXPORT_PRIVATE static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    JS_EXPORT_PRIVATE static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned);
    JS_EXPORT_PRIVATE static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    JS_EXPORT_PRIVATE static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    JS_EXPORT_PRIVATE static JSValue getPrototype(JSObject*, JSGlobalObject*);
    JS_EXPORT_PRIVATE static bool setPrototype(JSObject*, JSGlobalObject*, JSValue, bool shouldThrowIfCantSet);
    JS_EXPORT_PRIVATE static bool isExtensible(JSObject*, JSGlobalObject*);
    JS_EXPORT_PRIVATE static bool preventExtensions(JSObject*, JSGlobalObject*);

    RefPtr<const Wasm::RTT> m_rtt;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
