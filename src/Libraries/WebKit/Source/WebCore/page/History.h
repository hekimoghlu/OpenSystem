/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

#include "ExceptionOr.h"
#include "FrameLoaderTypes.h"
#include "JSValueInWrappedObject.h"
#include "LocalDOMWindowProperty.h"
#include "ScriptWrappable.h"
#include "SerializedScriptValue.h"
#include <wtf/WallTime.h>

namespace WebCore {

class Document;

class History final : public ScriptWrappable, public RefCounted<History>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(History);
public:
    static Ref<History> create(LocalDOMWindow& window) { return adoptRef(*new History(window)); }

    ExceptionOr<unsigned> length() const;

    enum class ScrollRestoration {
        Auto,
        Manual
    };

    ExceptionOr<ScrollRestoration> scrollRestoration() const;
    ExceptionOr<void> setScrollRestoration(ScrollRestoration);

    void setTotalStateObjectPayloadLimitOverride(std::optional<uint32_t> limit) { m_totalStateObjectPayloadLimitOverride = limit; }

    ExceptionOr<SerializedScriptValue*> state();
    JSValueInWrappedObject& cachedState();
    JSValueInWrappedObject& cachedStateForGC() { return m_cachedState; }

    ExceptionOr<void> back();
    ExceptionOr<void> forward();
    ExceptionOr<void> go(int);

    ExceptionOr<void> back(Document&);
    ExceptionOr<void> forward(Document&);
    ExceptionOr<void> go(Document&, int);

    bool isSameAsCurrentState(SerializedScriptValue*) const;

    ExceptionOr<void> pushState(RefPtr<SerializedScriptValue>&& data, const String&, const String& urlString);
    ExceptionOr<void> replaceState(RefPtr<SerializedScriptValue>&& data, const String&, const String& urlString);

private:
    explicit History(LocalDOMWindow&);

    ExceptionOr<void> stateObjectAdded(RefPtr<SerializedScriptValue>&&, const String& url, NavigationHistoryBehavior);
    ExceptionOr<void> updateAndCheckStateObjectQuota(const URL&, SerializedScriptValue*, NavigationHistoryBehavior);
    bool stateChanged() const;

    SerializedScriptValue* stateInternal() const;
    uint32_t totalStateObjectPayloadLimit() const;

    RefPtr<SerializedScriptValue> m_lastStateObjectRequested;
    JSValueInWrappedObject m_cachedState;

    unsigned m_currentStateObjectTimeSpanObjectsAdded { 0 };
    WallTime m_currentStateObjectTimeSpanStart;

    // For the main frame's History object to keep track of all state object usage.
    uint64_t m_totalStateObjectUsage { 0 };
    std::optional<uint32_t> m_totalStateObjectPayloadLimitOverride;

    // For each individual History object to keep track of the most recent state object added.
    uint64_t m_mostRecentStateObjectUsage { 0 };
};

inline ExceptionOr<void> History::pushState(RefPtr<SerializedScriptValue>&& data, const String&, const String& urlString)
{
    return stateObjectAdded(WTFMove(data), urlString, NavigationHistoryBehavior::Push);
}

inline ExceptionOr<void> History::replaceState(RefPtr<SerializedScriptValue>&& data, const String&, const String& urlString)
{
    return stateObjectAdded(WTFMove(data), urlString, NavigationHistoryBehavior::Replace);
}

} // namespace WebCore
