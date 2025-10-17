/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "config.h"
#include "WebErrors.h"

#include "APIError.h"
#include <WebCore/LocalizedStrings.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceResponse.h>

namespace WebKit {
using namespace WebCore;

ResourceError downloadNetworkError(const URL& failingURL, const String& localizedDescription)
{
    return ResourceError(API::Error::webKitDownloadErrorDomain(), API::Error::Download::Transport, failingURL, localizedDescription);
}

ResourceError downloadCancelledByUserError(const ResourceResponse& response)
{
    return ResourceError(API::Error::webKitDownloadErrorDomain(), API::Error::Download::CancelledByUser, response.url(), WEB_UI_STRING("User cancelled the download", "The download was cancelled by the user"));
}

ResourceError downloadDestinationError(const ResourceResponse& response, const String& localizedDescription)
{
    return ResourceError(API::Error::webKitDownloadErrorDomain(), API::Error::Download::Destination, response.url(), localizedDescription);
}

} // namespace WebKit
