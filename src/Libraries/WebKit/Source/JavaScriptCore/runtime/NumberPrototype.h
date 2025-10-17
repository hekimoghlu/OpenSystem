/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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

#include "NumberObject.h"

namespace JSC {

class NumberPrototype final : public NumberObject {
public:
    using Base = NumberObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static NumberPrototype* create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
    {
        NumberPrototype* prototype = new (NotNull, allocateCell<NumberPrototype>(vm)) NumberPrototype(vm, structure);
        prototype->finishCreation(vm, globalObject);
        return prototype;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    NumberPrototype(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(NumberPrototype, NumberObject);

JSC_DECLARE_HOST_FUNCTION(numberProtoFuncValueOf);
JSC_DECLARE_HOST_FUNCTION(numberProtoFuncToString);
JSString* int32ToString(VM&, int32_t value, int32_t radix);
JSString* int52ToString(VM&, int64_t value, int32_t radix);
JSString* numberToString(VM&, double value, int32_t radix);
String toStringWithRadix(double doubleValue, int32_t radix);
int32_t extractToStringRadixArgument(JSGlobalObject*, JSValue radixValue, ThrowScope&);

} // namespace JSC
