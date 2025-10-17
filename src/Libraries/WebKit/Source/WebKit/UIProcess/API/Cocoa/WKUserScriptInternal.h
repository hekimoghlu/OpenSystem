/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#import "WKUserScriptPrivate.h"

#import "APIUserScript.h"

namespace WebKit {

template<> struct WrapperTraits<API::UserScript> {
    using WrapperClass = WKUserScript;
};

}

namespace API {

inline WebCore::UserScriptInjectionTime toWebCoreUserScriptInjectionTime(WKUserScriptInjectionTime injectionTime)
{
    switch (injectionTime) {
    case WKUserScriptInjectionTimeAtDocumentStart:
        return WebCore::UserScriptInjectionTime::DocumentStart;
    case WKUserScriptInjectionTimeAtDocumentEnd:
        return WebCore::UserScriptInjectionTime::DocumentEnd;
    }

    ASSERT_NOT_REACHED();
    return WebCore::UserScriptInjectionTime::DocumentEnd;
}

inline WKUserScriptInjectionTime toWKUserScriptInjectionTime(WebCore::UserScriptInjectionTime injectionTime)
{
    switch (injectionTime) {
    case WebCore::UserScriptInjectionTime::DocumentStart:
        return WKUserScriptInjectionTimeAtDocumentStart;
    case WebCore::UserScriptInjectionTime::DocumentEnd:
        return WKUserScriptInjectionTimeAtDocumentEnd;
    }

    ASSERT_NOT_REACHED();
    return WKUserScriptInjectionTimeAtDocumentEnd;
}

}

@interface WKUserScript () <WKObject> {
@package
    API::ObjectStorage<API::UserScript> _userScript;
}
@end
