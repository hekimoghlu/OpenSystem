/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

#include "APIObject.h"
#include "StorageNamespaceIdentifier.h"
#include "WebPageGroupData.h"
#include <wtf/Ref.h>

namespace WebCore {
class PageGroup;
}

namespace WebKit {

class WebUserContentController;

class WebPageGroupProxy : public RefCounted<WebPageGroupProxy> {
public:
    static Ref<WebPageGroupProxy> create(const WebPageGroupData&);
    virtual ~WebPageGroupProxy();

    const String& identifier() const { return m_data.identifier; }
    PageGroupIdentifier pageGroupID() const { return m_data.pageGroupID; }
    // Namespace IDs for local storage namespaces are currently equivalent to web page group IDs.
    WebCore::PageGroup* corePageGroup() const;

private:
    WebPageGroupProxy(const WebPageGroupData&);

    WebPageGroupData m_data;
    WeakPtr<WebCore::PageGroup> m_pageGroup;
};

} // namespace WebKit
