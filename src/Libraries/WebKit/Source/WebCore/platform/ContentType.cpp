/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "ContentType.h"
#include "MIMETypeRegistry.h"
#include <wtf/JSONValues.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/URL.h>

namespace WebCore {

ContentType::ContentType(String&& contentType)
    : m_type(WTFMove(contentType))
{
}

ContentType::ContentType(const String& contentType)
    : m_type(contentType)
{
}

ContentType::ContentType(const String& contentType, bool typeWasInferredFromExtension)
    : m_type(contentType)
    , m_typeWasInferredFromExtension(typeWasInferredFromExtension)
{
}

ContentType ContentType::fromURL(const URL& url)
{
    ASSERT(isMainThread());

    auto lastPathComponent = url.lastPathComponent();
    size_t pos = lastPathComponent.reverseFind('.');
    if (pos != notFound) {
        auto extension = lastPathComponent.substring(pos + 1);
        String mediaType = MIMETypeRegistry::mediaMIMETypeForExtension(extension);
        if (!mediaType.isEmpty())
            return ContentType(WTFMove(mediaType), true);
    }
    return ContentType();
}

const String& ContentType::codecsParameter()
{
    static NeverDestroyed<String> codecs { "codecs"_s };
    return codecs;
}

const String& ContentType::profilesParameter()
{
    static NeverDestroyed<String> profiles { "profiles"_s };
    return profiles;
}

String ContentType::parameter(const String& parameterName) const
{
    // A MIME type can have one or more "param=value" after a semicolon, separated from each other by semicolons.

    // FIXME: This will ignore a quotation mark if it comes before the semicolon. Is that the desired behavior?
    auto semicolonPosition = m_type.find(';');
    if (semicolonPosition == notFound)
        return { };

    // FIXME: This matches parameters that have parameterName as a suffix; that is not the desired behavior.
    auto nameStart = m_type.findIgnoringASCIICase(parameterName, semicolonPosition + 1);
    if (nameStart == notFound)
        return { };

    auto equalSignPosition = m_type.find('=', nameStart + parameterName.length());
    if (equalSignPosition == notFound)
        return { };

    // FIXME: This skips over any characters that come before a quotation mark; that is not the desired behavior.
    auto quotePosition = m_type.find('"', equalSignPosition + 1);
    // FIXME: This does not work if there is an escaped quotation mark in the quoted string. Is that the desired behavior?
    auto secondQuotePosition = m_type.find('"', quotePosition + 1);
    size_t start;
    size_t end;
    if (quotePosition != notFound && secondQuotePosition != notFound) {
        start = quotePosition + 1;
        end = secondQuotePosition;
    } else {
        // FIXME: If there is only one quotation mark, this will treat it as part of the string; that is not the desired behavior.
        start = equalSignPosition + 1;
        end = m_type.find(';', start);
    }
    return StringView { m_type }.substring(start, end - start).trim(isASCIIWhitespace<UChar>).toString();
}

String ContentType::containerType() const
{
    // Strip parameters that come after a semicolon.
    // FIXME: This will ignore a quotation mark if it comes before the semicolon. Is that the desired behavior?
    return m_type.left(m_type.find(';')).trim(isASCIIWhitespace);
}

static inline Vector<String> splitParameters(StringView parametersView)
{
    Vector<String> result;
    for (auto view : parametersView.split(','))
        result.append(view.trim(isASCIIWhitespace<UChar>).toString());
    return result;
}

Vector<String> ContentType::codecs() const
{
    return splitParameters(parameter(codecsParameter()));
}

Vector<String> ContentType::profiles() const
{
    return splitParameters(parameter(profilesParameter()));
}

String ContentType::toJSONString() const
{
    auto object = JSON::Object::create();

    object->setString("containerType"_s, containerType());

    auto codecs = codecsParameter();
    if (!codecs.isEmpty())
        object->setString("codecs"_s, codecs);

    auto profiles = profilesParameter();
    if (!profiles.isEmpty())
        object->setString("profiles"_s, profiles);

    return object->toJSONString();
}

} // namespace WebCore
