/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include "JSObject.h"

namespace JSC {

template<typename ViewClass>
class JSGenericTypedArrayViewPrototype final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSGenericTypedArrayViewPrototype, Base);
        return &vm.plainObjectSpace();
    }

    static JSGenericTypedArrayViewPrototype* create(
        VM&, JSGlobalObject*, Structure*);

    // FIXME: We should fix the warnings for extern-template in JSObject template classes: https://bugs.webkit.org/show_bug.cgi?id=161979
    IGNORE_CLANG_WARNINGS_BEGIN("undefined-var-template")
    DECLARE_INFO;
    IGNORE_CLANG_WARNINGS_END
    
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

private:
    JSGenericTypedArrayViewPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JSC_DECLARE_HOST_FUNCTION(uint8ArrayPrototypeSetFromBase64);
JSC_DECLARE_HOST_FUNCTION(uint8ArrayPrototypeSetFromHex);
JSC_DECLARE_HOST_FUNCTION(uint8ArrayPrototypeToBase64);
JSC_DECLARE_HOST_FUNCTION(uint8ArrayPrototypeToHex);

} // namespace JSC
