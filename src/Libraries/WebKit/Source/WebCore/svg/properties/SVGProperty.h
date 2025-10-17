/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#include "SVGPropertyOwner.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class SVGPropertyAccess : uint8_t { ReadWrite, ReadOnly };
enum class SVGPropertyState : uint8_t { Clean, Dirty };

class SVGProperty : public ThreadSafeRefCounted<SVGProperty> {
public:
    virtual ~SVGProperty() = default;

    // Managing the relationship with the owner.
    bool isAttached() const { return m_owner; }
    virtual void attach(SVGPropertyOwner* owner, SVGPropertyAccess access)
    {
        ASSERT(!m_owner);
        ASSERT(m_state == SVGPropertyState::Clean);
        m_owner = owner;
        m_access = access;
    }

    virtual void detach()
    {
        m_owner = nullptr;
        m_access = SVGPropertyAccess::ReadWrite;
        m_state = SVGPropertyState::Clean;
    }

    void reattach(SVGPropertyOwner* owner, SVGPropertyAccess access)
    {
        ASSERT_UNUSED(owner, owner == m_owner);
        m_access = access;
        m_state = SVGPropertyState::Clean;
    }

    const SVGElement* contextElement() const
    {
        if (!m_owner)
            return nullptr;
        return m_owner->attributeContextElement();
    }

    void commitChange()
    {
        if (!m_owner)
            return;
        m_owner->commitPropertyChange(this);
    }

    // DOM access.
    SVGPropertyAccess access() const { return m_access; }
    bool isReadOnly() const { return m_access == SVGPropertyAccess::ReadOnly; }

    // Synchronizing the SVG attribute and its reflection here.
    bool isDirty() const { return m_state == SVGPropertyState::Dirty; }
    void setDirty() { m_state = SVGPropertyState::Dirty; }
    std::optional<String> synchronize()
    {
        if (m_state == SVGPropertyState::Clean)
            return std::nullopt;
        m_state = SVGPropertyState::Clean;
        return valueAsString();
    }

    // This is used when calling setAttribute().
    virtual String valueAsString() const { return emptyString(); }

protected:
    SVGProperty(SVGPropertyOwner* owner = nullptr, SVGPropertyAccess access = SVGPropertyAccess::ReadWrite)
        : m_owner(owner)
        , m_access(access)
    {
    }

    SVGPropertyOwner* m_owner { nullptr };
    SVGPropertyAccess m_access { SVGPropertyAccess::ReadWrite };
    SVGPropertyState m_state { SVGPropertyState::Clean };
};

} // namespace WebCore
