/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include "AppHighlight.h"
#include "AppHighlightRangeData.h"
#include "EventTarget.h"
#include <wtf/Forward.h>
#include <wtf/MonotonicTime.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

#if ENABLE(APP_HIGHLIGHTS)

class Document;
class FragmentedSharedBuffer;
class StaticRange;
class Highlight;

enum class RestoreWithTextSearch : bool { No, Yes };

enum class ScrollToHighlight : bool { No, Yes };

class AppHighlightStorage final {
    WTF_MAKE_TZONE_ALLOCATED(AppHighlightStorage);
public:
    AppHighlightStorage(Document&);
    ~AppHighlightStorage();

    WEBCORE_EXPORT void storeAppHighlight(Ref<StaticRange>&&, CompletionHandler<void(AppHighlight&&)>&&);
    WEBCORE_EXPORT void restoreAndScrollToAppHighlight(Ref<FragmentedSharedBuffer>&&, ScrollToHighlight);
    void restoreUnrestoredAppHighlights();

    bool shouldRestoreHighlights(MonotonicTime timestamp);

    bool hasUnrestoredHighlights() const { return m_unrestoredHighlights.size() || m_unrestoredScrollHighlight; }

private:
    bool attemptToRestoreHighlightAndScroll(AppHighlightRangeData&, ScrollToHighlight);
    
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    MonotonicTime m_timeAtLastRangeSearch;
    Vector<AppHighlightRangeData> m_unrestoredHighlights;
    std::optional<AppHighlightRangeData> m_unrestoredScrollHighlight;
};

#endif

} // namespace WebCore
