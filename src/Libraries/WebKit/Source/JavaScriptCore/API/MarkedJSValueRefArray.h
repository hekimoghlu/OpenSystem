/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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

#include "APICast.h"
#include "ArgList.h"
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/UniqueArray.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class MarkedJSValueRefArray final : public BasicRawSentinelNode<MarkedJSValueRefArray> {
    WTF_MAKE_NONCOPYABLE(MarkedJSValueRefArray);
    WTF_MAKE_NONMOVABLE(MarkedJSValueRefArray);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    static constexpr size_t inlineCapacity = MarkedArgumentBuffer::inlineCapacity;

    JS_EXPORT_PRIVATE MarkedJSValueRefArray(JSGlobalContextRef, unsigned);
    JS_EXPORT_PRIVATE ~MarkedJSValueRefArray();

    size_t size() const { return m_size; }
    bool isEmpty() const { return !m_size; }

    JSValueRef& operator[](unsigned index) { return data()[index]; }

    const JSValueRef* data() const
    {
        return const_cast<MarkedJSValueRefArray*>(this)->data();
    }

    JSValueRef* data()
    {
        if (m_buffer)
            return m_buffer.get();
        return m_inlineBuffer;
    }

    template<typename Visitor> void visitAggregate(Visitor&);

private:
    unsigned m_size;
    JSValueRef m_inlineBuffer[inlineCapacity] { };
    UniqueArray<JSValueRef> m_buffer;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
