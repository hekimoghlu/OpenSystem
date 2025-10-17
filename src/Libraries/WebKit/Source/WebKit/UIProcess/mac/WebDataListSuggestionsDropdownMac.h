/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

#if USE(APPKIT)

#import "WebDataListSuggestionsDropdown.h"
#import <wtf/RetainPtr.h>

OBJC_CLASS WKDataListSuggestionsController;

namespace WebKit {

class WebDataListSuggestionsDropdownMac final : public WebDataListSuggestionsDropdown {
public:
    static Ref<WebDataListSuggestionsDropdownMac> create(WebPageProxy&, NSView *);
    ~WebDataListSuggestionsDropdownMac();

    void didSelectOption(const String&);

private:
    WebDataListSuggestionsDropdownMac(WebPageProxy&, NSView *);

    void show(WebCore::DataListSuggestionInformation&&) final;
    void handleKeydownWithIdentifier(const String&) final;
    void close() final;

    void selectOption();

    NSView *m_view;
    RetainPtr<WKDataListSuggestionsController> m_dropdownUI;
};

} // namespace WebKit

#endif // USE(APPKIT)
