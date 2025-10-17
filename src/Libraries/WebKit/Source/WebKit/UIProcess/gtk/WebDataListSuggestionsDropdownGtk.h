/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

#include "WebDataListSuggestionsDropdown.h"

typedef struct _GtkTreePath GtkTreePath;
typedef struct _GtkTreeView GtkTreeView;
typedef struct _GtkTreeViewColumn GtkTreeViewColumn;

namespace WebKit {

class WebPageProxy;

class WebDataListSuggestionsDropdownGtk final : public WebDataListSuggestionsDropdown {
public:
    static Ref<WebDataListSuggestionsDropdownGtk> create(GtkWidget* webView, WebPageProxy& page)
    {
        return adoptRef(*new WebDataListSuggestionsDropdownGtk(webView, page));
    }

    ~WebDataListSuggestionsDropdownGtk();

private:
    WebDataListSuggestionsDropdownGtk(GtkWidget*, WebPageProxy&);

    void show(WebCore::DataListSuggestionInformation&&) final;
    void handleKeydownWithIdentifier(const String&) final;
    void close() final;

    static void treeViewRowActivatedCallback(GtkTreeView*, GtkTreePath*, GtkTreeViewColumn*, WebDataListSuggestionsDropdownGtk*);
    void didSelectOption(const String&);

    GtkWidget* m_webView { nullptr };
    GtkWidget* m_popup { nullptr };
    GtkWidget* m_treeView { nullptr };
};

} // namespace WebKit
