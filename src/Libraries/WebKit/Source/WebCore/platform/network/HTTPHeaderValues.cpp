/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#include "HTTPHeaderValues.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

namespace HTTPHeaderValues {

const String& textPlainContentType()
{
    static NeverDestroyed<const String> contentType(MAKE_STATIC_STRING_IMPL("text/plain;charset=UTF-8"));
    return contentType;
}

const String& formURLEncodedContentType()
{
    static NeverDestroyed<const String> contentType(MAKE_STATIC_STRING_IMPL("application/x-www-form-urlencoded;charset=UTF-8"));
    return contentType;
}

const String& applicationJSONContentType()
{
    // The default encoding is UTF-8: https://www.ietf.org/rfc/rfc4627.txt.
    static NeverDestroyed<const String> contentType(MAKE_STATIC_STRING_IMPL("application/json"));
    return contentType;
}

const String& noCache()
{
    static NeverDestroyed<const String> value(MAKE_STATIC_STRING_IMPL("no-cache"));
    return value;
}

const String& maxAge0()
{
    static NeverDestroyed<const String> value(MAKE_STATIC_STRING_IMPL("max-age=0"));
    return value;
}

}

}
