/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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

#include "JSFunction.h"
#include "MathCommon.h"

namespace JSC {

class NumberPrototype;
class GetterSetter;

class NumberConstructor final : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static NumberConstructor* create(VM&, Structure*, NumberPrototype*);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static bool isIntegerImpl(JSValue value)
    {
        return value.isInt32() || (value.isDouble() && isInteger(value.asDouble()));
    }

private:
    NumberConstructor(VM&, NativeExecutable*, JSGlobalObject*, Structure*);
    void finishCreation(VM&, NumberPrototype*);
};
static_assert(sizeof(NumberConstructor) == sizeof(JSFunction), "Allocate NumberConstructor in JSFunction IsoSubspace");

} // namespace JSC
