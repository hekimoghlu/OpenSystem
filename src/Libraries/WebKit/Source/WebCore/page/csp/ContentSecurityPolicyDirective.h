/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentSecurityPolicyDirectiveList;

class ContentSecurityPolicyDirective {
    WTF_MAKE_TZONE_ALLOCATED(ContentSecurityPolicyDirective);
public:
    ContentSecurityPolicyDirective(const ContentSecurityPolicyDirectiveList& directiveList, const String& name, const String& value)
        : m_name(name)
        , m_text(makeString(name, ' ', value))
        , m_directiveList(directiveList)
    {
    }

    virtual ~ContentSecurityPolicyDirective() = 0;

    const String& name() const { return m_name; }
    const String& text() const { return m_text; }
    virtual const String& nameForReporting() const { return m_name; }

    const ContentSecurityPolicyDirectiveList& directiveList() const { return m_directiveList; }

    bool isDefaultSrc() const;

private:
    String m_name;
    String m_text;
    const ContentSecurityPolicyDirectiveList& m_directiveList;
};

} // namespace WebCore
