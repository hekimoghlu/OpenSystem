/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "IntlSegmenterPrototype.h"

#include "IntlSegmenter.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlSegmenterPrototypeFuncSegment);
static JSC_DECLARE_HOST_FUNCTION(intlSegmenterPrototypeFuncResolvedOptions);

}

#include "IntlSegmenterPrototype.lut.h"

namespace JSC {

const ClassInfo IntlSegmenterPrototype::s_info = { "Intl.Segmenter"_s, &Base::s_info, &segmenterPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlSegmenterPrototype) };

/* Source for IntlSegmenterPrototype.lut.h
@begin segmenterPrototypeTable
  segment          intlSegmenterPrototypeFuncSegment            DontEnum|Function 1
  resolvedOptions  intlSegmenterPrototypeFuncResolvedOptions    DontEnum|Function 0
@end
*/

IntlSegmenterPrototype* IntlSegmenterPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlSegmenterPrototype>(vm)) IntlSegmenterPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlSegmenterPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlSegmenterPrototype::IntlSegmenterPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlSegmenterPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-intl-segmenter/#sec-intl.segmenter.prototype.segment
JSC_DEFINE_HOST_FUNCTION(intlSegmenterPrototypeFuncSegment, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* segmenter = jsDynamicCast<IntlSegmenter*>(callFrame->thisValue());
    if (UNLIKELY(!segmenter))
        return throwVMTypeError(globalObject, scope, "Intl.Segmenter.prototype.segment called on value that's not a Segmenter"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(segmenter->segment(globalObject, callFrame->argument(0))));
}

// https://tc39.es/proposal-intl-segmenter/#sec-Intl.Segmenter.prototype.resolvedOptions
JSC_DEFINE_HOST_FUNCTION(intlSegmenterPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* segmenter = jsDynamicCast<IntlSegmenter*>(callFrame->thisValue());
    if (UNLIKELY(!segmenter))
        return throwVMTypeError(globalObject, scope, "Intl.Segmenter.prototype.resolvedOptions called on value that's not a Segmenter"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(segmenter->resolvedOptions(globalObject)));
}

} // namespace JSC
