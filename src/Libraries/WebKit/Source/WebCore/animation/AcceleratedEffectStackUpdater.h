/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include <wtf/HashSet.h>
#include <wtf/Seconds.h>

namespace WebCore {

class Document;
class Element;
struct Styleable;

class AcceleratedEffectStackUpdater {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Animation);
public:
    AcceleratedEffectStackUpdater(Document&);

    void updateEffectStacks();
    void updateEffectStackForTarget(const Styleable&);

    Seconds timeOrigin() const { return m_timeOrigin; }

protected:

private:
    using HashedStyleable = std::pair<Element*, std::optional<Style::PseudoElementIdentifier>>;
    HashSet<HashedStyleable> m_targetsPendingUpdate;
    Seconds m_timeOrigin;
};

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
