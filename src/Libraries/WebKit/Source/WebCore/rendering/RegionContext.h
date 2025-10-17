/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

#include "AffineTransform.h"
#include "IntRect.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class Path;

class RegionContext : public CanMakeCheckedPtr<RegionContext> {
    WTF_MAKE_TZONE_ALLOCATED(RegionContext);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RegionContext);
public:
    virtual ~RegionContext() = default;

    void pushTransform(const AffineTransform&);
    void popTransform();

    void pushClip(const IntRect&);
    void pushClip(const Path&);
    void popClip();

    virtual bool isEventRegionContext() const { return false; }
    virtual bool isAccessibilityRegionContext() const { return false; }

protected:
    Vector<AffineTransform> m_transformStack;
    Vector<IntRect> m_clipStack;

}; // class RegionContext

class RegionContextStateSaver {
public:
    RegionContextStateSaver(RegionContext* context)
        : m_context(context)
    {
    }

    ~RegionContextStateSaver()
    {
        if (!m_context)
            return;

        if (m_pushedClip)
            m_context->popClip();
    }

    void pushClip(const IntRect& clipRect)
    {
        ASSERT(!m_pushedClip);
        if (m_context)
            m_context->pushClip(clipRect);
        m_pushedClip = true;
    }

    void pushClip(const Path& path)
    {
        ASSERT(!m_pushedClip);
        if (m_context)
            m_context->pushClip(path);
        m_pushedClip = true;
    }

    RegionContext* context() const { return const_cast<RegionContext*>(m_context.get()); }

private:
    CheckedPtr<RegionContext> m_context;
    bool m_pushedClip { false };
}; // class RegionContextStateSaver

} // namespace WebCore
