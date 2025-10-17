/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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

#include <wtf/CheckedRef.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class FrameDestructionObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::FrameDestructionObserver> : std::true_type { };
}

namespace WebCore {

class LocalFrame;

class FrameDestructionObserver : public CanMakeWeakPtr<FrameDestructionObserver> {
public:
    WEBCORE_EXPORT explicit FrameDestructionObserver(LocalFrame*);

    WEBCORE_EXPORT virtual void frameDestroyed();
    WEBCORE_EXPORT virtual void willDetachPage();

    inline LocalFrame* frame() const; // Defined in FrameDestructionObserverInlines.h.
    inline RefPtr<LocalFrame> protectedFrame() const; // Defined in FrameDestructionObserverInlines.h.

protected:
    WEBCORE_EXPORT virtual ~FrameDestructionObserver();
    WEBCORE_EXPORT void observeFrame(LocalFrame*);

    WeakPtr<LocalFrame> m_frame;
};

} // namespace WebCore
