/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include "DOMQuadInit.h"
#include "ScriptWrappable.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class DOMPoint;
class DOMRect;
struct DOMPointInit;
struct DOMRectInit;

class DOMQuad : public ScriptWrappable, public RefCounted<DOMQuad> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMQuad);
public:
    static Ref<DOMQuad> create(const DOMPointInit& p1, const DOMPointInit& p2, const DOMPointInit& p3, const DOMPointInit& p4) { return adoptRef(*new DOMQuad(p1, p2, p3, p4)); }
    static Ref<DOMQuad> fromRect(const DOMRectInit& init) { return adoptRef(*new DOMQuad(init)); }
    static Ref<DOMQuad> fromQuad(const DOMQuadInit& init) { return create(init.p1, init.p2, init.p3, init.p4); }

    const DOMPoint& p1() const { return m_p1; }
    const DOMPoint& p2() const { return m_p2; }
    const DOMPoint& p3() const { return m_p3; }
    const DOMPoint& p4() const { return m_p4; }

    Ref<DOMRect> getBounds() const;

private:
    DOMQuad(const DOMPointInit&, const DOMPointInit&, const DOMPointInit&, const DOMPointInit&);
    explicit DOMQuad(const DOMRectInit&);
    
    Ref<DOMPoint> m_p1;
    Ref<DOMPoint> m_p2;
    Ref<DOMPoint> m_p3;
    Ref<DOMPoint> m_p4;
};

}
