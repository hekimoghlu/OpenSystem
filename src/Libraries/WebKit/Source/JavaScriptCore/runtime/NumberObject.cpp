/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "NumberObject.h"

#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(NumberObject);

const ClassInfo NumberObject::s_info = { "Number"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(NumberObject) };

NumberObject::NumberObject(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

#if ASSERT_ENABLED
void NumberObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    ASSERT(type() == NumberObjectType);
}
#endif

NumberObject* constructNumber(JSGlobalObject* globalObject, JSValue number)
{
    NumberObject* object = NumberObject::create(globalObject->vm(), globalObject->numberObjectStructure());
    object->setInternalValue(globalObject->vm(), number);
    return object;
}

} // namespace JSC
