/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#include "SameSiteInfo.h"

#include "HTTPParsers.h"
#include "ResourceRequest.h"

namespace WebCore {

SameSiteInfo SameSiteInfo::create(const ResourceRequest& request, IsForDOMCookieAccess isForDOMAccess)
{
    // SameSite=strict cookies should be returned in document.cookie.
    // See https://github.com/httpwg/http-extensions/issues/769
    // and https://github.com/httpwg/http-extensions/pull/1428/files.
    auto isSameSite = request.isSameSite() || (isForDOMAccess == IsForDOMCookieAccess::Yes && request.isTopSite());
    return { isSameSite, request.isTopSite(), isSafeMethod(request.httpMethod()) };
}

} // namespace WebCore
