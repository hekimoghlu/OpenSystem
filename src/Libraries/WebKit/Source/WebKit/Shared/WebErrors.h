/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#include <wtf/Forward.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {

WebCore::ResourceError cancelledError(const WebCore::ResourceRequest&);
WebCore::ResourceError blockedError(const WebCore::ResourceRequest&);
WebCore::ResourceError blockedByContentBlockerError(const WebCore::ResourceRequest&);
WebCore::ResourceError cannotShowURLError(const WebCore::ResourceRequest&);
WebCore::ResourceError wasBlockedByRestrictionsError(const WebCore::ResourceRequest&);
WebCore::ResourceError interruptedForPolicyChangeError(const WebCore::ResourceRequest&);
WebCore::ResourceError ftpDisabledError(const WebCore::ResourceRequest&);
WebCore::ResourceError failedCustomProtocolSyncLoad(const WebCore::ResourceRequest&);
#if ENABLE(CONTENT_FILTERING)
WebCore::ResourceError blockedByContentFilterError(const WebCore::ResourceRequest&);
#endif
WebCore::ResourceError cannotShowMIMETypeError(const WebCore::ResourceResponse&);
WebCore::ResourceError fileDoesNotExistError(const WebCore::ResourceResponse&);
WebCore::ResourceError httpsUpgradeRedirectLoopError(const WebCore::ResourceRequest&);
WebCore::ResourceError httpNavigationWithHTTPSOnlyError(const WebCore::ResourceRequest&);
WebCore::ResourceError pluginWillHandleLoadError(const WebCore::ResourceResponse&);

#if USE(SOUP)
WebCore::ResourceError downloadNetworkError(const URL&, const WTF::String&);
WebCore::ResourceError downloadCancelledByUserError(const WebCore::ResourceResponse&);
WebCore::ResourceError downloadDestinationError(const WebCore::ResourceResponse&, const WTF::String&);
#endif

WebCore::ResourceError decodeError(const URL&);

#if PLATFORM(GTK)
WebCore::ResourceError invalidPageRangeToPrint(const URL&);
#endif

} // namespace WebKit
