/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include "BrandedStructure.h"

#include "JSCInlines.h"

namespace JSC {

BrandedStructure::BrandedStructure(VM& vm, Structure* previous, UniquedStringImpl* brandUid)
    : Structure(vm, previous)
    , m_brand(brandUid)
    , m_parentBrand(previous->isBrandedStructure() ? previous : nullptr, WriteBarrierEarlyInit)
{
    this->setIsBrandedStructure(true);
}

BrandedStructure::BrandedStructure(VM& vm, BrandedStructure* previous)
    : Structure(vm, previous)
    , m_brand(previous->m_brand)
    , m_parentBrand(previous->m_parentBrand.get(), WriteBarrierEarlyInit)
{
    this->setIsBrandedStructure(true);
}

Structure* BrandedStructure::create(VM& vm, Structure* previous, UniquedStringImpl* brandUid, DeferredStructureTransitionWatchpointFire* deferred)
{
    ASSERT(vm.structureStructure);
    BrandedStructure* newStructure = new (NotNull, allocateCell<BrandedStructure>(vm)) BrandedStructure(vm, previous, brandUid);
    newStructure->finishCreation(vm, previous, deferred);
    ASSERT(newStructure->type() == StructureType);
    return newStructure;
}

} // namespace JSC
