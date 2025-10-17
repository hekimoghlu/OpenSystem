/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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

#include "ContentSecurityPolicyDirective.h"
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContentSecurityPolicyDirectiveList;
enum class AllowTrustedTypePolicy : uint8_t;

class ContentSecurityPolicyTrustedTypesDirective : public ContentSecurityPolicyDirective {
    WTF_MAKE_TZONE_ALLOCATED(ContentSecurityPolicyTrustedTypesDirective);
public:
    ContentSecurityPolicyTrustedTypesDirective(const ContentSecurityPolicyDirectiveList&, const String& name, const String& value);

    bool allows(const String& value, bool isDuplicate, AllowTrustedTypePolicy&) const;

private:
    void parse(const String&);

    bool m_allowDuplicates { false };
    bool m_allowAny { false };
    UncheckedKeyHashSet<String> m_list;
};

} // namespace WebCore
