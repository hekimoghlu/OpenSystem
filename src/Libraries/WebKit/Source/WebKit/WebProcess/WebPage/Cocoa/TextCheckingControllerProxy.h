/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#if ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)

#include "Connection.h"
#include "EditingRange.h"
#include "MessageReceiver.h"
#include <WebCore/SimpleRange.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebCore {
class VisiblePosition;
struct AttributedString;
}

namespace WebKit {

class WebPage;

class TextCheckingControllerProxy : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(TextCheckingControllerProxy);
public:
    TextCheckingControllerProxy(WebPage&);
    ~TextCheckingControllerProxy();

    void ref() const final;
    void deref() const final;

    static WebCore::AttributedString annotatedSubstringBetweenPositions(const WebCore::VisiblePosition&, const WebCore::VisiblePosition&);

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    struct RangeAndOffset {
        WebCore::SimpleRange range;
        size_t locationInRoot;    
    };
    std::optional<RangeAndOffset> rangeAndOffsetRelativeToSelection(int64_t offset, uint64_t length);

    // Message handlers.
    void replaceRelativeToSelection(const WebCore::AttributedString&, int64_t selectionOffset, uint64_t length, uint64_t relativeReplacementLocation, uint64_t relativeReplacementLength);
    void removeAnnotationRelativeToSelection(const String& annotationName, int64_t selectionOffset, uint64_t length);

    WeakRef<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(PLATFORM_DRIVEN_TEXT_CHECKING)
