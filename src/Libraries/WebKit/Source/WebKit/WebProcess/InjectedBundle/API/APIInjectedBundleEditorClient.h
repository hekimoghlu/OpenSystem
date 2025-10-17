/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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

#include <WebCore/EditorInsertAction.h>
#include <WebCore/TextAffinity.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SharedBuffer;
class DocumentFragment;
class Node;
class StyleProperties;
struct SimpleRange;
}

namespace WebKit {
class WebPage;
}

namespace API {

namespace InjectedBundle {

class EditorClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(EditorClient);
public:
    virtual ~EditorClient() { }

    virtual bool shouldBeginEditing(WebKit::WebPage&, const WebCore::SimpleRange&) { return true; }
    virtual bool shouldEndEditing(WebKit::WebPage&, const WebCore::SimpleRange&) { return true; }
    virtual bool shouldInsertNode(WebKit::WebPage&, WebCore::Node&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) { return true; }
    virtual bool shouldInsertText(WebKit::WebPage&, const WTF::String&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) { return true; }
    virtual bool shouldDeleteRange(WebKit::WebPage&, const std::optional<WebCore::SimpleRange>&) { return true; }
    virtual bool shouldChangeSelectedRange(WebKit::WebPage&, const std::optional<WebCore::SimpleRange>&, const std::optional<WebCore::SimpleRange>&, WebCore::Affinity, bool) { return true; }
    virtual bool shouldApplyStyle(WebKit::WebPage&, const WebCore::StyleProperties&, const std::optional<WebCore::SimpleRange>&) { return true; }
    virtual void didBeginEditing(WebKit::WebPage&, const WTF::String&) { }
    virtual void didEndEditing(WebKit::WebPage&, const WTF::String&) { }
    virtual void didChange(WebKit::WebPage&, const WTF::String&) { }
    virtual void didChangeSelection(WebKit::WebPage&, const WTF::String&) { }
    virtual void willWriteToPasteboard(WebKit::WebPage&, const std::optional<WebCore::SimpleRange>&) { }
    virtual void getPasteboardDataForRange(WebKit::WebPage&, const std::optional<WebCore::SimpleRange>&, Vector<WTF::String>&, Vector<RefPtr<WebCore::SharedBuffer>>&) { }
    virtual void didWriteToPasteboard(WebKit::WebPage&) { }
    virtual bool performTwoStepDrop(WebKit::WebPage&, WebCore::DocumentFragment&, const WebCore::SimpleRange&, bool) { return false; }
};

} // namespace InjectedBundle

}
