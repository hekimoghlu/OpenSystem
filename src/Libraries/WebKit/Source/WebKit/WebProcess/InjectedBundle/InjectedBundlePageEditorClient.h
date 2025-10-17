/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include "APIClient.h"
#include "APIInjectedBundleEditorClient.h"
#include "WKBundlePageEditorClient.h"
#include <wtf/TZoneMalloc.h>

namespace API {
template<> struct ClientTraits<WKBundlePageEditorClientBase> {
    typedef std::tuple<WKBundlePageEditorClientV0, WKBundlePageEditorClientV1> Versions;
};
}

namespace WebCore {
class CSSStyleDeclaration;
class DocumentFragment;
class Node;
struct SimpleRange;
}

namespace WebKit {

class WebFrame;
class WebPage;

class InjectedBundlePageEditorClient final : public API::Client<WKBundlePageEditorClientBase>, public API::InjectedBundle::EditorClient {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageEditorClient);
public:
    explicit InjectedBundlePageEditorClient(const WKBundlePageEditorClientBase&);

private:
    bool shouldBeginEditing(WebPage&, const WebCore::SimpleRange&) final;
    bool shouldEndEditing(WebPage&, const WebCore::SimpleRange&) final;
    bool shouldInsertNode(WebPage&, WebCore::Node&, const std::optional<WebCore::SimpleRange>& rangeToReplace, WebCore::EditorInsertAction) final;
    bool shouldInsertText(WebPage&, const String&, const std::optional<WebCore::SimpleRange>& rangeToReplace, WebCore::EditorInsertAction) final;
    bool shouldDeleteRange(WebPage&, const std::optional<WebCore::SimpleRange>&) final;
    bool shouldChangeSelectedRange(WebPage&, const std::optional<WebCore::SimpleRange>& fromRange, const std::optional<WebCore::SimpleRange>& toRange, WebCore::Affinity, bool stillSelecting) final;
    bool shouldApplyStyle(WebPage&, const WebCore::StyleProperties&, const std::optional<WebCore::SimpleRange>&) final;
    void didBeginEditing(WebPage&, const String& notificationName) final;
    void didEndEditing(WebPage&, const String& notificationName) final;
    void didChange(WebPage&, const String& notificationName) final;
    void didChangeSelection(WebPage&, const String& notificationName) final;
    void willWriteToPasteboard(WebPage&, const std::optional<WebCore::SimpleRange>&) final;
    void getPasteboardDataForRange(WebPage&, const std::optional<WebCore::SimpleRange>&, Vector<String>& pasteboardTypes, Vector<RefPtr<WebCore::SharedBuffer>>& pasteboardData) final;
    void didWriteToPasteboard(WebPage&) final;
    bool performTwoStepDrop(WebPage&, WebCore::DocumentFragment&, const WebCore::SimpleRange& destination, bool isMove) final;
};

} // namespace WebKit
