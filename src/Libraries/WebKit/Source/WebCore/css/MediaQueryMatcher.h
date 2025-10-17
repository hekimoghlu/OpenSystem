/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

#include <memory>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class MediaQueryList;
class RenderStyle;
class WeakPtrImplWithEventTargetData;

namespace MQ {
struct MediaQuery;
using MediaQueryList = Vector<MediaQuery>;
}

// MediaQueryMatcher class is responsible for evaluating the queries whenever it
// is needed and dispatch "change" event on MediaQueryLists if the corresponding
// query has changed. MediaQueryLists are invoked in the order in which they were added.

class MediaQueryMatcher final : public RefCounted<MediaQueryMatcher> {
public:
    static Ref<MediaQueryMatcher> create(Document& document) { return adoptRef(*new MediaQueryMatcher(document)); }
    ~MediaQueryMatcher();

    void documentDestroyed();
    void addMediaQueryList(MediaQueryList&);
    void removeMediaQueryList(MediaQueryList&);

    RefPtr<MediaQueryList> matchMedia(const String&);

    unsigned evaluationRound() const { return m_evaluationRound; }

    enum class EventMode : uint8_t { Schedule, DispatchNow };
    void evaluateAll(EventMode);

    bool evaluate(const MQ::MediaQueryList&);

    AtomString mediaType() const;

private:
    explicit MediaQueryMatcher(Document&);
    std::unique_ptr<RenderStyle> documentElementUserAgentStyle() const;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    Vector<WeakPtr<MediaQueryList, WeakPtrImplWithEventTargetData>> m_mediaQueryLists;

    // This value is incremented at style selector changes.
    // It is used to avoid evaluating queries more then once and to make sure
    // that a media query result change is notified exactly once.
    unsigned m_evaluationRound { 1 };
};

} // namespace WebCore
