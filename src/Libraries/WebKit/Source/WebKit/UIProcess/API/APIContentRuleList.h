/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#include "NetworkCacheData.h"
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebCompiledContentRuleList;
}

namespace API {

class ContentRuleList final : public ObjectImpl<Object::Type::ContentRuleList> {
public:
#if ENABLE(CONTENT_EXTENSIONS)
    static Ref<ContentRuleList> create(Ref<WebKit::WebCompiledContentRuleList>&& contentRuleList, WebKit::NetworkCache::Data&& mappedFile)
    {
        return adoptRef(*new ContentRuleList(WTFMove(contentRuleList), WTFMove(mappedFile)));
    }

    ContentRuleList(Ref<WebKit::WebCompiledContentRuleList>&&, WebKit::NetworkCache::Data&&);
    virtual ~ContentRuleList();

    const WTF::String& name() const;
    const WebKit::WebCompiledContentRuleList& compiledRuleList() const { return m_compiledRuleList.get(); }
    
    static bool supportsRegularExpression(const WTF::String&);
    static std::error_code parseRuleList(const WTF::String&);

private:
    Ref<WebKit::WebCompiledContentRuleList> m_compiledRuleList;
    WebKit::NetworkCache::Data m_mappedFile;
#endif // ENABLE(CONTENT_EXTENSIONS)
};

} // namespace API
