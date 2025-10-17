/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>

// WARNING! WARNING! WARNING!
//
// The user agent is ludicrously fragile. The most innocent change can
// and will break websites. Read the git log for this file and
// WebCore/platform/glib/UserAgentGLib.cpp carefully before changing
// user agent construction. You have been warned.

namespace WebCore {

static String getSystemSoftwareName()
{
#if HAS_GETENV_NP
    char buf[32];
    if (!getenv_np("SYSTEM_SOFTWARE_NAME", buf, sizeof(buf)))
        return String::fromUTF8(buf);
#endif
    return "PlayStation"_s;
}

static String getSystemSoftwareVersion()
{
#if HAS_GETENV_NP
    char buf[32];
    if (!getenv_np("SYSTEM_SOFTWARE_VERSION", buf, sizeof(buf)))
        return String::fromUTF8(buf);
#endif
    return "0.00"_s;
}

static constexpr ASCIILiteral versionForUAString()
{
    // https://bugs.webkit.org/show_bug.cgi?id=180365
    return "605.1.15"_s;
}

static String standardUserAgentStatic()
{
    // Version/X is mandatory *before* Safari/X to be a valid Safari UA. See
    // https://bugs.webkit.org/show_bug.cgi?id=133403 for details.
    static NeverDestroyed<String> uaStatic(makeString("Mozilla/5.0 (PlayStation; "_s, getSystemSoftwareName(), '/', getSystemSoftwareVersion(), ") AppleWebKit/"_s, versionForUAString(), " (KHTML, like Gecko) "_s, "Version/17.0 Safari/"_s, versionForUAString()));
    return uaStatic;
}

String standardUserAgent(const String& applicationName, const String& applicationVersion)
{
    // Create a default user agent string with a liberal interpretation of
    // https://developer.mozilla.org/en-US/docs/User_Agent_Strings_Reference
    //
    // Forming a functional user agent is really difficult. We must mention Safari, because some
    // sites check for that when detecting WebKit browsers. Additionally some sites assume that
    // browsers that are "Safari" but not running on OS X are the Safari iOS browser. Getting this
    // wrong can cause sites to load the wrong JavaScript, CSS, or custom fonts. In some cases
    // sites won't load resources at all.
    if (applicationName.isEmpty())
        return standardUserAgentStatic();

    return makeString(standardUserAgentStatic(), ' ', applicationName, '/', applicationVersion.isEmpty() ? versionForUAString() : applicationVersion);
}

String standardUserAgentForURL(const URL&)
{
    return standardUserAgentStatic();
}

} // namespace WebCore
