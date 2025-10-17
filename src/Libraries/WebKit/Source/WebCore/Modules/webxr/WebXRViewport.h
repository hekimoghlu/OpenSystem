/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#if ENABLE(WEBXR)

#include "IntRect.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WebXRViewport : public RefCounted<WebXRViewport> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRViewport);
public:
    static Ref<WebXRViewport> create(const IntRect&);

    int x() const { return m_viewport.x(); }
    int y() const { return m_viewport.y(); }
    int width() const { return m_viewport.width(); }
    int height() const { return m_viewport.height(); }
    IntRect rect() { return m_viewport; }

    void updateViewport(const IntRect& viewport) { m_viewport = viewport; }

private:
    explicit WebXRViewport(const IntRect&);

    IntRect m_viewport;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
