/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#include "GPUQuerySet.h"
#include "GPUQuerySetDescriptor.h"

namespace WebCore {

GPUQuerySet::GPUQuerySet(Ref<WebGPU::QuerySet>&& backing, const GPUQuerySetDescriptor& descriptor)
    : m_backing(WTFMove(backing))
    , m_descriptor(descriptor)
{
}

String GPUQuerySet::label() const
{
    return m_backing->label();
}

void GPUQuerySet::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

void GPUQuerySet::destroy()
{
    m_backing->destroy();
}

GPUQueryType GPUQuerySet::type() const
{
    return m_descriptor.type;
}

GPUSize32Out GPUQuerySet::count() const
{
    return m_descriptor.count;
}

}
