/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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

#include <WebCore/ContentRuleListResults.h>

namespace API {

class ContentRuleListAction final : public ObjectImpl<Object::Type::ContentRuleListAction> {
public:
#if ENABLE(CONTENT_EXTENSIONS)
    static Ref<ContentRuleListAction> create(WebCore::ContentRuleListResults::Result&&);
    virtual ~ContentRuleListAction();

    bool blockedLoad() const;
    bool madeHTTPS() const;
    bool blockedCookies() const;
    bool redirected() const;
    bool modifiedHeaders() const;
    const Vector<WTF::String>& notifications() const;

private:
    ContentRuleListAction(WebCore::ContentRuleListResults::Result&&);

    WebCore::ContentRuleListResults::Result m_result;
#endif // ENABLE(CONTENT_EXTENSIONS)
};
    
} // namespace API
