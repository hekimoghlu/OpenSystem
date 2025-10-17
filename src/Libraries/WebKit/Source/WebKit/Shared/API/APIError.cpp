/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#include "APIError.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace API {

const WTF::String& Error::webKitErrorDomain()
{
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitErrorDomain"));
    return webKitErrorDomainString;
}

const WTF::String& Error::webKitNetworkErrorDomain()
{
#if USE(GLIB)
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitNetworkError"));
    return webKitErrorDomainString;
#else
    return webKitErrorDomain();
#endif
}

const WTF::String& Error::webKitPolicyErrorDomain()
{
#if USE(GLIB)
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitPolicyError"));
    return webKitErrorDomainString;
#else
    return webKitErrorDomain();
#endif
}

const WTF::String& Error::webKitPluginErrorDomain()
{
#if USE(GLIB)
#if ENABLE(2022_GLIB_API)
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitMediaError"));
#else
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitPluginError"));
#endif
    return webKitErrorDomainString;
#else
    return webKitErrorDomain();
#endif
}

#if USE(SOUP)
const WTF::String& Error::webKitDownloadErrorDomain()
{
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitDownloadError"));
    return webKitErrorDomainString;
}
#endif

#if PLATFORM(GTK)
const WTF::String& Error::webKitPrintErrorDomain()
{
    static NeverDestroyed<WTF::String> webKitErrorDomainString(MAKE_STATIC_STRING_IMPL("WebKitPrintError"));
    return webKitErrorDomainString;
}
#endif

} // namespace WebKit
