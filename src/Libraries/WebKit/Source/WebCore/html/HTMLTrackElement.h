/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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

#if ENABLE(VIDEO)

#include "ActiveDOMObject.h"
#include "HTMLElement.h"
#include "TextTrackClient.h"

namespace WebCore {

class HTMLMediaElement;
class LoadableTextTrack;

class HTMLTrackElement final : public HTMLElement, public ActiveDOMObject, public TextTrackClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTrackElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTrackElement);
public:
    static Ref<HTMLTrackElement> create(const QualifiedName&, Document&);

    // ActiveDOMObject.
    void ref() const final { HTMLElement::ref(); }
    void deref() const final { HTMLElement::deref(); }

    using HTMLElement::scriptExecutionContext;

    USING_CAN_MAKE_WEAKPTR(HTMLElement);

    const AtomString& kind();
    void setKind(const AtomString&);

    const AtomString& srclang() const;
    const AtomString& label() const;
    bool isDefault() const;

    enum ReadyState { NONE = 0, LOADING = 1, LOADED = 2, TRACK_ERROR = 3 };
    ReadyState readyState() const;
    void setReadyState(ReadyState);

    TextTrack& track();

    void scheduleLoad();

    enum LoadStatus { Failure, Success };
    void didCompleteLoad(LoadStatus);

    RefPtr<HTMLMediaElement> mediaElement() const;
    const AtomString& mediaElementCrossOriginAttribute() const;

    void scheduleTask(Function<void()>&&);

private:
    HTMLTrackElement(const QualifiedName&, Document&);
    virtual ~HTMLTrackElement();

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    bool isURLAttribute(const Attribute&) const final;

    // EventTarget.
    void eventListenersDidChange() final;

    // TextTrackClient
    void textTrackModeChanged(TextTrack&) final;

    bool canLoadURL(const URL&);

    Ref<LoadableTextTrack> m_track;
    bool m_loadPending { false };
    bool m_hasRelevantLoadEventsListener { false };
};

}

#endif
