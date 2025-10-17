/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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

#include "InternalFunction.h"
#include "IntlObject.h"

namespace JSC {

class IntlDisplayNamesPrototype;

class IntlDisplayNamesConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static IntlDisplayNamesConstructor* create(VM&, Structure*, IntlDisplayNamesPrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    IntlDisplayNamesConstructor(VM&, Structure*);
    void finishCreation(VM&, IntlDisplayNamesPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(IntlDisplayNamesConstructor, InternalFunction);

} // namespace JSC
