/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#include "MarkedJSValueRefArray.h"

#include "JSCJSValue.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

MarkedJSValueRefArray::MarkedJSValueRefArray(JSGlobalContextRef context, unsigned size)
    : m_size(size)
{
    if (m_size > MarkedArgumentBuffer::inlineCapacity) {
        m_buffer = makeUniqueArray<JSValueRef>(m_size);
        toJS(context)->vm().heap.addMarkedJSValueRefArray(this);
        ASSERT(isOnList());
    }
}

MarkedJSValueRefArray::~MarkedJSValueRefArray()
{
    if (isOnList())
        remove();
}

template<typename Visitor>
void MarkedJSValueRefArray::visitAggregate(Visitor& visitor)
{
    JSValueRef* buffer = data();
    for (unsigned index = 0; index < m_size; ++index) {
        JSValueRef value = buffer[index];
#if !CPU(ADDRESS64)
        JSCell* jsCell = reinterpret_cast<JSCell*>(const_cast<OpaqueJSValue*>(value));
        if (!jsCell)
            continue;
        visitor.appendUnbarriered(jsCell); // We should mark the wrapper itself to keep JSValueRef live.
#else
        visitor.appendUnbarriered(std::bit_cast<JSValue>(value));
#endif
    }
}

template void MarkedJSValueRefArray::visitAggregate(AbstractSlotVisitor&);
template void MarkedJSValueRefArray::visitAggregate(SlotVisitor&);

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
