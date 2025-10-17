/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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

#if ENABLE(VIDEO)
#include "APIObject.h"
#include <WebCore/CaptionUserPreferences.h>
#include <wtf/Ref.h>

namespace API {

class CaptionUserPreferencesTestingModeToken : public API::ObjectImpl<API::Object::Type::CaptionUserPreferencesTestingModeToken> {
public:
    static Ref<CaptionUserPreferencesTestingModeToken> create(WebCore::CaptionUserPreferences& preferences)
    {
        return adoptRef(*new CaptionUserPreferencesTestingModeToken(*new WebCore::CaptionUserPreferencesTestingModeToken(preferences)));
    }
private:
    CaptionUserPreferencesTestingModeToken(WebCore::CaptionUserPreferencesTestingModeToken& token)
        : m_token(token)
    {
    }

    WebCore::CaptionUserPreferencesTestingModeToken m_token;
};

}

#endif // ENABLE(VIDEO)
