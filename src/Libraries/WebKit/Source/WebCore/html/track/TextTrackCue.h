/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "DocumentFragment.h"
#include "HTMLElement.h"
#include <wtf/JSONValues.h>
#include <wtf/MediaTime.h>

namespace WebCore {

class SpeechSynthesis;
class TextTrack;
class TextTrackCue;

class TextTrackCueBox : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextTrackCueBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextTrackCueBox);
public:
    static Ref<TextTrackCueBox> create(Document&, TextTrackCue&);

    TextTrackCue* getCue() const;
    virtual void applyCSSProperties() { }

protected:
    void initialize();

    TextTrackCueBox(Document&, TextTrackCue&);
    ~TextTrackCueBox() { }

private:

    WeakPtr<TextTrackCue, WeakPtrImplWithEventTargetData> m_cue;
};

class TextTrackCue : public RefCounted<TextTrackCue>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextTrackCue);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static ExceptionOr<Ref<TextTrackCue>> create(Document&, double start, double end, DocumentFragment&);

    void didMoveToNewDocument(Document&);

    TextTrack* track() const;
    RefPtr<TextTrack> protectedTrack() const;
    void setTrack(TextTrack*);

    const AtomString& id() const { return m_id; }
    void setId(const AtomString&);

    double startTime() const { return startMediaTime().toDouble(); }
    void setStartTime(double);

    double endTime() const { return endMediaTime().toDouble(); }
    void setEndTime(double);

    bool pauseOnExit() const { return m_pauseOnExit; }
    void setPauseOnExit(bool);

    MediaTime startMediaTime() const { return m_startTime; }
    void setStartTime(const MediaTime&);

    MediaTime endMediaTime() const { return m_endTime; }
    void setEndTime(const MediaTime&);

    bool isActive() const;
    virtual void setIsActive(bool);

    virtual bool isOrderedBefore(const TextTrackCue*) const;
    virtual bool isPositionedAbove(const TextTrackCue* cue) const { return isOrderedBefore(cue); }

    bool hasEquivalentStartTime(const TextTrackCue&) const;

    enum CueType { Generic, Data, ConvertedToWebVTT, WebVTT };
    virtual CueType cueType() const { return CueType::Generic; }
    virtual bool isRenderable() const;

    enum CueMatchRules { MatchAllFields, IgnoreDuration };
    bool isEqual(const TextTrackCue&, CueMatchRules) const;

    void willChange();
    virtual void didChange(bool = false);

    virtual RefPtr<TextTrackCueBox> getDisplayTree();
    virtual void removeDisplayTree();

    virtual RefPtr<DocumentFragment> getCueAsHTML();
    virtual const String& text() const { return emptyString(); }

    String toJSONString() const;

    virtual void recalculateStyles() { m_displayTreeNeedsUpdate = true; }
    virtual void setFontSize(int fontSize, bool important);
    virtual void updateDisplayTree(const MediaTime&) { }

    unsigned cueIndex() const;

    using SpeakCueCompletionHandler = Function<void(const TextTrackCue&)>;
    virtual void prepareToSpeak(SpeechSynthesis&, double, double, SpeakCueCompletionHandler&&) { }
    virtual void beginSpeaking() { }
    virtual void pauseSpeaking() { }
    virtual void cancelSpeaking() { }

    virtual bool cueContentsMatch(const TextTrackCue&) const;

protected:
    TextTrackCue(Document&, const MediaTime& start, const MediaTime& end);

    Document* document() const;

    virtual void toJSON(JSON::Object&) const;

private:
    TextTrackCue(Document&, const MediaTime& start, const MediaTime& end, Ref<DocumentFragment>&&);

    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    using EventTarget::dispatchEvent;
    void dispatchEvent(Event&) final;
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::TextTrackCue; }
    ScriptExecutionContext* scriptExecutionContext() const final;

    void rebuildDisplayTree();

    AtomString m_id;
    MediaTime m_startTime;
    MediaTime m_endTime;
    int m_processingCueChanges { 0 };

    WeakPtr<TextTrack, WeakPtrImplWithEventTargetData> m_track;

    RefPtr<DocumentFragment> m_cueNode;
    RefPtr<TextTrackCueBox> m_displayTree;

    int m_fontSize { 0 };
    bool m_fontSizeIsImportant { false };

    bool m_isActive { false };
    bool m_pauseOnExit { false };
    bool m_displayTreeNeedsUpdate { true };
};

#ifndef NDEBUG
TextStream& operator<<(TextStream&, const TextTrackCue&);
#endif

} // namespace WebCore

namespace WTF {

template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::TextTrackCue> {
    static String toString(const WebCore::TextTrackCue& cue) { return cue.toJSONString(); }
};

}

#endif
