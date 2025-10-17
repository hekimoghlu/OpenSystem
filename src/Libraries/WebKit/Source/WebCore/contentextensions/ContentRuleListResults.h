/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionActions.h"
#include <wtf/KeyValuePair.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
    
struct ContentRuleListResults {
    struct Result {
        bool blockedLoad { false };
        bool madeHTTPS { false };
        bool blockedCookies { false };
        bool modifiedHeaders { false };
        bool redirected { false };
        Vector<String> notifications;
        
        bool shouldNotifyApplication() const
        {
            return blockedLoad
                || madeHTTPS
                || blockedCookies
                || modifiedHeaders
                || redirected
                || !notifications.isEmpty();
        }
    };
    struct Summary {
        bool blockedLoad { false };
        bool madeHTTPS { false };
        bool blockedCookies { false };
        bool hasNotifications { false };
        // Remaining fields currently aren't serialized as they aren't required by _WKContentRuleListAction
        Vector<ContentExtensions::ModifyHeadersAction> modifyHeadersActions { };
        Vector<std::pair<ContentExtensions::RedirectAction, URL>> redirectActions { };
    };
    using ContentRuleListIdentifier = String;

    Summary summary;
    Vector<std::pair<ContentRuleListIdentifier, Result>> results;
    
    bool shouldNotifyApplication() const
    {
        return summary.blockedLoad
            || summary.madeHTTPS
            || summary.blockedCookies
            || !summary.modifyHeadersActions.isEmpty()
            || !summary.redirectActions.isEmpty()
            || summary.hasNotifications;
    }
};

}

#endif // ENABLE(CONTENT_EXTENSIONS)
