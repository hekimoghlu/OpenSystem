/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "UserAgent.h"

#include "SystemInfo.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

String standardUserAgent(const String& applicationName, const String& applicationVersion)
{
    auto version = applicationName.isEmpty() ? emptyString() : applicationVersion;
    return makeString("Mozilla/5.0 ("_s, windowsVersionForUAString(), ") AppleWebKit/605.1.15 (KHTML, like Gecko)"_s,
        applicationName.isEmpty() ? ""_s : " "_s, applicationName, version.isEmpty() ? ""_s : "/"_s, version);
}

} // namespace WebCore
