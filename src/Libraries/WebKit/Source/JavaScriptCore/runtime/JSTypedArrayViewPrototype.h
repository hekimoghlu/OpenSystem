/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

class JSTypedArrayViewPrototype final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTypedArrayViewPrototype, Base);
        return &vm.plainObjectSpace();
    }

    static JSTypedArrayViewPrototype* create(VM&, JSGlobalObject*, Structure*);

    DECLARE_INFO;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

private:
    JSTypedArrayViewPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncIsTypedArrayView);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncIsSharedTypedArrayView);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncIsResizableOrGrowableSharedTypedArrayView);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncTypedArrayFromFast);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncIsDetached);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncLength);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncContentType);
JSC_DECLARE_HOST_FUNCTION(typedArrayViewPrivateFuncGetOriginalConstructor);

} // namespace JSC
