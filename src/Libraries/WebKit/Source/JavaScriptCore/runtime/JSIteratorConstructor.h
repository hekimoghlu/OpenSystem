/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

namespace JSC {

class JSIteratorPrototype;

// https://tc39.es/proposal-iterator-helpers/#sec-iterator-constructor
class JSIteratorConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    static JSIteratorConstructor* create(VM&, JSGlobalObject*, Structure*, JSIteratorPrototype*);

    DECLARE_INFO;
    DECLARE_VISIT_CHILDREN;
private:
    JSIteratorConstructor(VM&, Structure*);

    void finishCreation(VM&, JSGlobalObject*, JSIteratorPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSIteratorConstructor, InternalFunction);

} // namespace JSC
