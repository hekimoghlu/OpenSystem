/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include "IntrinsicGetterAccessCase.h"

#if ENABLE(JIT)

#include "HeapInlines.h"
#include "JSCellInlines.h"
#include "JSTypedArrays.h"

namespace JSC {

IntrinsicGetterAccessCase::IntrinsicGetterAccessCase(VM& vm, JSCell* owner, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure, const ObjectPropertyConditionSet& conditionSet, JSFunction* intrinsicFunction, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
    : Base(vm, owner, IntrinsicGetter, identifier, offset, structure, conditionSet, WTFMove(prototypeAccessChain))
{
    m_intrinsicFunction.set(vm, owner, intrinsicFunction);
}

Ref<AccessCase> IntrinsicGetterAccessCase::create(VM& vm, JSCell* owner, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure, const ObjectPropertyConditionSet& conditionSet, JSFunction* intrinsicFunction, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
{
    return adoptRef(*new IntrinsicGetterAccessCase(vm, owner, identifier, offset, structure, conditionSet, intrinsicFunction, WTFMove(prototypeAccessChain)));
}

bool IntrinsicGetterAccessCase::doesCalls() const
{
    switch (intrinsic()) {
    case TypedArrayByteOffsetIntrinsic:
    case TypedArrayByteLengthIntrinsic:
    case TypedArrayLengthIntrinsic:
    case DataViewByteLengthIntrinsic:
        return isResizableOrGrowableSharedTypedArrayIncludingDataView(structure()->classInfoForCells());
    default:
        return false;
    }
    return false;
}

} // namespace JSC

#endif // ENABLE(JIT)
