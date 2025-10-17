/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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

#include "IntersectionObserver.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Document;
class Element;
class HTMLIFrameElement;

class LazyLoadFrameObserver {
    WTF_MAKE_TZONE_ALLOCATED(LazyLoadFrameObserver);
public:
    LazyLoadFrameObserver(HTMLIFrameElement&);

    void observe(const AtomString& frameURL, const ReferrerPolicy&);
    void unobserve();

    AtomString frameURL() const { return m_frameURL; }
    ReferrerPolicy referrerPolicy() const { return m_referrerPolicy; }

private:
    IntersectionObserver* intersectionObserver(Document&);
    bool isObserved(Element&) const;

    WeakRef<HTMLIFrameElement, WeakPtrImplWithEventTargetData> m_element;
    AtomString m_frameURL;
    ReferrerPolicy m_referrerPolicy;
    RefPtr<IntersectionObserver> m_observer;
};

} // namespace
