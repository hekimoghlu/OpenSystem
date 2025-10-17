/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include "AcceleratedEffect.h"
#include "AcceleratedEffectValues.h"

namespace WebCore {

using AcceleratedEffects = Vector<Ref<AcceleratedEffect>>;

class WEBCORE_EXPORT AcceleratedEffectStack : public RefCounted<AcceleratedEffectStack> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(AcceleratedEffectStack, WEBCORE_EXPORT);
public:
    static Ref<AcceleratedEffectStack> create();

    bool hasEffects() const;
    const AcceleratedEffects& primaryLayerEffects() const { return m_primaryLayerEffects; }
    const AcceleratedEffects& backdropLayerEffects() const { return m_backdropLayerEffects; }
    virtual void setEffects(AcceleratedEffects&&);

    const AcceleratedEffectValues& baseValues() { return m_baseValues; }
    void setBaseValues(AcceleratedEffectValues&&);

    virtual ~AcceleratedEffectStack() = default;

protected:
    WEBCORE_EXPORT explicit AcceleratedEffectStack();

    AcceleratedEffectValues m_baseValues;
    AcceleratedEffects m_primaryLayerEffects;
    AcceleratedEffects m_backdropLayerEffects;
};

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
