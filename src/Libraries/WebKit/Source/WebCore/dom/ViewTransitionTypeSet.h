/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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

#include "Element.h"
#include "JSDOMSetLike.h"
#include <wtf/ListHashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class ViewTransitionTypeSet : public RefCounted<ViewTransitionTypeSet> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ViewTransitionTypeSet);

public:
    static Ref<ViewTransitionTypeSet> create(Document& document, Vector<AtomString>&& initialActiveTypes)
    {
        return adoptRef(*new ViewTransitionTypeSet(document, WTFMove(initialActiveTypes)));
    }

    void initializeSetLike(DOMSetAdapter&) const;

    void clearFromSetLike();
    void addToSetLike(const AtomString&);
    bool removeFromSetLike(const AtomString&);

    bool hasType(const AtomString&) const;

private:
    ViewTransitionTypeSet(Document&, Vector<AtomString>&&);

    ListHashSet<AtomString> m_typeSet;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
};

}
