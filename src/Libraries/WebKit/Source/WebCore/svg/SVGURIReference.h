/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#include "Document.h"
#include "QualifiedName.h"
#include "SVGPropertyOwnerRegistry.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGElement;

class SVGURIReference {
    WTF_MAKE_TZONE_ALLOCATED(SVGURIReference);
    WTF_MAKE_NONCOPYABLE(SVGURIReference);
public:
    virtual ~SVGURIReference() = default;

    void parseAttribute(const QualifiedName&, const AtomString&);

    static AtomString fragmentIdentifierFromIRIString(const String&, const Document&);

    struct TargetElementResult {
        RefPtr<Element> element;
        AtomString identifier;
    };
    static TargetElementResult targetElementFromIRIString(const String&, const TreeScope&, RefPtr<Document> externalDocument = nullptr);

    static bool isExternalURIReference(const String& uri, const Document& document)
    {
        // Fragment-only URIs are always internal
        if (uri.startsWith('#'))
            return false;

        // If the URI matches our documents URL, we're dealing with a local reference.
        URL url = document.completeURL(uri);
        ASSERT(!url.protocolIsData());
        return !equalIgnoringFragmentIdentifier(url, document.url());
    }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGURIReference>;

    String href() const { return m_href->currentValue(); }
    SVGAnimatedString& hrefAnimated() { return m_href; }

protected:
    SVGURIReference(SVGElement* contextElement);

    static bool isKnownAttribute(const QualifiedName& attributeName);

    virtual bool haveFiredLoadEvent() const { return false; }
    virtual void setHaveFiredLoadEvent(bool) { }
    virtual bool errorOccurred() const { return false; }
    virtual void setErrorOccurred(bool) { }

    bool haveLoadedRequiredResources() const;
    void dispatchLoadEvent();

private:
    SVGElement& contextElement() const;

    Ref<SVGAnimatedString> m_href;
};

} // namespace WebCore
