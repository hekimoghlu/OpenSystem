/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class Document;
class Element;
class SVGElement;
class SVGFontFaceElement;
class SVGResourcesCache;
class SVGSMILElement;
class SVGSVGElement;
class SVGUseElement;
class WeakPtrImplWithEventTargetData;

class SVGDocumentExtensions final : public CanMakeCheckedPtr<SVGDocumentExtensions> {
    WTF_MAKE_TZONE_ALLOCATED(SVGDocumentExtensions);
    WTF_MAKE_NONCOPYABLE(SVGDocumentExtensions);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGDocumentExtensions);
public:
    explicit SVGDocumentExtensions(Document&);
    ~SVGDocumentExtensions();

    void addTimeContainer(SVGSVGElement&);
    void removeTimeContainer(SVGSVGElement&);
    WEBCORE_EXPORT Vector<Ref<SVGSVGElement>> allSVGSVGElements() const;

    void startAnimations();
    void pauseAnimations();
    void unpauseAnimations();
    void dispatchLoadEventToOutermostSVGElements();
    bool areAnimationsPaused() const { return m_areAnimationsPaused; }

    void reportWarning(const String&);
    void reportError(const String&);

    SVGResourcesCache& resourcesCache() { return *m_resourcesCache; }

    void addElementToRebuild(SVGElement&);
    void removeElementToRebuild(SVGElement&);
    void rebuildElements();
    void clearTargetDependencies(SVGElement&);
    void rebuildAllElementReferencesForTarget(SVGElement&);

    const WeakHashSet<SVGFontFaceElement, WeakPtrImplWithEventTargetData>& svgFontFaceElements() const { return m_svgFontFaceElements; }
    void registerSVGFontFaceElement(SVGFontFaceElement&);
    void unregisterSVGFontFaceElement(SVGFontFaceElement&);

private:
    Ref<Document> protectedDocument() const;

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    WeakHashSet<SVGSVGElement, WeakPtrImplWithEventTargetData> m_timeContainers; // For SVG 1.2 support this will need to be made more general.
    WeakHashSet<SVGFontFaceElement, WeakPtrImplWithEventTargetData> m_svgFontFaceElements;
    std::unique_ptr<SVGResourcesCache> m_resourcesCache;

    Vector<Ref<SVGElement>> m_rebuildElements;
    bool m_areAnimationsPaused;

};

} // namespace WebCore
