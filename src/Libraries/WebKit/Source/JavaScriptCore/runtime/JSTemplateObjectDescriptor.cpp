/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "JSTemplateObjectDescriptor.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "VM.h"

namespace JSC {

const ClassInfo JSTemplateObjectDescriptor::s_info = { "TemplateObjectDescriptor"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(JSTemplateObjectDescriptor) };


JSTemplateObjectDescriptor::JSTemplateObjectDescriptor(VM& vm, Ref<TemplateObjectDescriptor>&& descriptor, int endOffset)
    : Base(vm, vm.templateObjectDescriptorStructure.get())
    , m_descriptor(WTFMove(descriptor))
    , m_endOffset(endOffset)
{
}

JSTemplateObjectDescriptor* JSTemplateObjectDescriptor::create(VM& vm, Ref<TemplateObjectDescriptor>&& descriptor, int endOffset)
{
    JSTemplateObjectDescriptor* result = new (NotNull, allocateCell<JSTemplateObjectDescriptor>(vm)) JSTemplateObjectDescriptor(vm, WTFMove(descriptor), endOffset);
    result->finishCreation(vm);
    return result;
}

void JSTemplateObjectDescriptor::destroy(JSCell* cell)
{
    static_cast<JSTemplateObjectDescriptor*>(cell)->JSTemplateObjectDescriptor::~JSTemplateObjectDescriptor();
}

JSArray* JSTemplateObjectDescriptor::createTemplateObject(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    unsigned count = descriptor().cookedStrings().size();
    JSArray* templateObject = constructEmptyArray(globalObject, nullptr, count);
    RETURN_IF_EXCEPTION(scope, nullptr);
    JSArray* rawObject = constructEmptyArray(globalObject, nullptr, count);
    RETURN_IF_EXCEPTION(scope, nullptr);

    for (unsigned index = 0; index < count; ++index) {
        auto cooked = descriptor().cookedStrings()[index];
        if (cooked)
            templateObject->putDirectIndex(globalObject, index, jsString(vm, cooked.value()), PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete, PutDirectIndexLikePutDirect);
        else
            templateObject->putDirectIndex(globalObject, index, jsUndefined(), PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete, PutDirectIndexLikePutDirect);
        RETURN_IF_EXCEPTION(scope, nullptr);

        rawObject->putDirectIndex(globalObject, index, jsString(vm, descriptor().rawStrings()[index]), PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete, PutDirectIndexLikePutDirect);
        RETURN_IF_EXCEPTION(scope, nullptr);
    }

    objectConstructorFreeze(globalObject, rawObject);
    RETURN_IF_EXCEPTION(scope, nullptr);

    templateObject->putDirect(vm, vm.propertyNames->raw, rawObject, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);

    objectConstructorFreeze(globalObject, templateObject);
    scope.assertNoExceptionExceptTermination();

    return templateObject;
}

} // namespace JSC
