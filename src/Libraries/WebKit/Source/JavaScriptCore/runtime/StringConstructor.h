/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

class StringPrototype;
class GetterSetter;

class StringConstructor final : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static StringConstructor* create(VM&, Structure*, StringPrototype*);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    StringConstructor(VM&, NativeExecutable*, JSGlobalObject*, Structure*);
    void finishCreation(VM&, StringPrototype*);
};
static_assert(sizeof(StringConstructor) == sizeof(JSFunction), "Allocate StringConstructor in JSFunction IsoSubspace");

JSString* stringFromCharCode(JSGlobalObject*, int32_t);
JSString* stringConstructor(JSGlobalObject*, JSValue);

} // namespace JSC
