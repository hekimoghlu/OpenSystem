/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#include "Color.h"
#include "ControlFactory.h"
#include "PlatformControl.h"
#include "StyleAppearance.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class FloatRect;
class GraphicsContext;
class ControlFactory;

class ControlPart : public ThreadSafeRefCounted<ControlPart> {
public:
    virtual ~ControlPart() = default;

    StyleAppearance type() const { return m_type; }

    WEBCORE_EXPORT ControlFactory& controlFactory() const;
    void setOverrideControlFactory(RefPtr<ControlFactory>&& controlFactory) { m_overrideControlFactory = WTFMove(controlFactory); }

    FloatSize sizeForBounds(const FloatRect& bounds, const ControlStyle&);
    FloatRect rectForBounds(const FloatRect& bounds, const ControlStyle&);
    void draw(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) const;

protected:
    WEBCORE_EXPORT ControlPart(StyleAppearance);

    PlatformControl* platformControl() const;
    virtual std::unique_ptr<PlatformControl> createPlatformControl() = 0;

    const StyleAppearance m_type;

    mutable std::unique_ptr<PlatformControl> m_platformControl;
    RefPtr<ControlFactory> m_controlFactory;
    RefPtr<ControlFactory> m_overrideControlFactory;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CONTROL_PART(PartName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PartName##Part) \
    static bool isType(const WebCore::ControlPart& part) { return part.type() == WebCore::StyleAppearance::PartName; } \
SPECIALIZE_TYPE_TRAITS_END()
