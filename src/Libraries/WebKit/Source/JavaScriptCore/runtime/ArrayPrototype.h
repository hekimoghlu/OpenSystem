/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

#include "JSArray.h"

namespace JSC {

class ArrayPrototype final : public JSArray {
public:
    using Base = JSArray;

    enum class SpeciesWatchpointStatus {
        Uninitialized,
        Initialized,
        Fired
    };

    static ArrayPrototype* create(VM&, JSGlobalObject*, Structure*);
        
    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    ArrayPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(ArrayPrototype, ArrayPrototype::Base);

JSC_DECLARE_HOST_FUNCTION(arrayProtoFuncToString);
JSC_DECLARE_HOST_FUNCTION(arrayProtoFuncValues);
JSC_DECLARE_HOST_FUNCTION(arrayProtoPrivateFuncAppendMemcpy);
JSC_DECLARE_HOST_FUNCTION(arrayProtoPrivateFuncFromFastFillWithUndefined);
JSC_DECLARE_HOST_FUNCTION(arrayProtoPrivateFuncFromFastFillWithEmpty);

} // namespace JSC
