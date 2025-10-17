/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include <WebCore/DataListSuggestionPicker.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class DataListSuggestionsClient;
}

namespace WebKit {

class WebPage;

class WebDataListSuggestionPicker final : public WebCore::DataListSuggestionPicker, public RefCounted<WebDataListSuggestionPicker> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DataListSuggestionPicker);
public:
    static Ref<WebDataListSuggestionPicker> create(WebPage& page, WebCore::DataListSuggestionsClient& client)
    {
        return adoptRef(*new WebDataListSuggestionPicker(page, client));
    }

    ~WebDataListSuggestionPicker();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void didSelectOption(const String&);
    void didCloseSuggestions();

private:
    WebDataListSuggestionPicker(WebPage&, WebCore::DataListSuggestionsClient&);

    void handleKeydownWithIdentifier(const String&) final;
    void displayWithActivationType(WebCore::DataListSuggestionActivationType) final;
    void close() final;
    void detach() final;

    CheckedPtr<WebCore::DataListSuggestionsClient> m_client;
    WeakPtr<WebPage> m_page;
};

} // namespace WebKit
