/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

namespace JSC {

class BooleanPrototype;
class GetterSetter;

class BooleanConstructor final : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    static BooleanConstructor* create(VM&, Structure*, BooleanPrototype*);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    BooleanConstructor(VM&, NativeExecutable*, JSGlobalObject*, Structure*);
    void finishCreation(VM&, BooleanPrototype*);
};
static_assert(sizeof(BooleanConstructor) == sizeof(JSFunction), "Allocate BooleanConstructor in JSFunction IsoSubspace");

JSObject* constructBooleanFromImmediateBoolean(JSGlobalObject*, JSValue);

} // namespace JSC
