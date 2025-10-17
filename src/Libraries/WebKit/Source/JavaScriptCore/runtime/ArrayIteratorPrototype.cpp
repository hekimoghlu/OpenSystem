/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "config.h"
#include "ArrayIteratorPrototype.h"

#include "JSCBuiltins.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo ArrayIteratorPrototype::s_info = { "Array Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ArrayIteratorPrototype) };

void ArrayIteratorPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->next, arrayIteratorPrototypeNextCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

} // namespace JSC
