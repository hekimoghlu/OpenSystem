/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

#include "Animation.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class AnimationList : public RefCounted<AnimationList> {
public:
    static Ref<AnimationList> create() { return adoptRef(*new AnimationList); }

    Ref<AnimationList> copy() const { return adoptRef(*new AnimationList(*this, CopyBehavior::Clone)); }
    Ref<AnimationList> shallowCopy() const { return adoptRef(*new AnimationList(*this, CopyBehavior::Reference)); }

    void fillUnsetProperties();
    bool operator==(const AnimationList&) const;
    
    size_t size() const { return m_animations.size(); }
    bool isEmpty() const { return m_animations.isEmpty(); }
    
    void resize(size_t n) { m_animations.resize(n); }
    void remove(size_t i) { m_animations.remove(i); }
    void append(Ref<Animation>&& animation) { m_animations.append(WTFMove(animation)); }

    Animation& animation(size_t i) { return m_animations[i].get(); }
    const Animation& animation(size_t i) const { return m_animations[i].get(); }

    auto begin() const { return m_animations.begin(); }
    auto end() const { return m_animations.end(); }

    using const_reverse_iterator = Vector<Ref<Animation>>::const_reverse_iterator;
    const_reverse_iterator rbegin() const { return m_animations.rbegin(); }
    const_reverse_iterator rend() const { return m_animations.rend(); }

private:
    AnimationList();

    enum class CopyBehavior : uint8_t { Clone, Reference };
    AnimationList(const AnimationList&, CopyBehavior);

    AnimationList& operator=(const AnimationList&);

    Vector<Ref<Animation>, 0, CrashOnOverflow, 0> m_animations;
};    

WTF::TextStream& operator<<(WTF::TextStream&, const AnimationList&);

} // namespace WebCore
