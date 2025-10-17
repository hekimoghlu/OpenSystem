/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

enum class Mode {
    Rfc2045,
    MimeSniff
};
WEBCORE_EXPORT bool isValidContentType(const String&, Mode = Mode::MimeSniff);

// FIXME: add support for comments.
class ParsedContentType {
public:
    WEBCORE_EXPORT static std::optional<ParsedContentType> create(const String&, Mode = Mode::MimeSniff);
    ParsedContentType(ParsedContentType&&) = default;

    String mimeType() const { return m_mimeType; }
    String charset() const;
    void setCharset(String&&);

    // Note that in the case of multiple values for the same name, the last value is returned.
    String parameterValueForName(const String&) const;
    size_t parameterCount() const;

    WEBCORE_EXPORT String serialize() const;

private:
    ParsedContentType(const String&);
    ParsedContentType(const ParsedContentType&) = delete;
    ParsedContentType& operator=(const ParsedContentType&) = delete;
    bool parseContentType(Mode);
    void setContentType(String&&, Mode);
    void setContentTypeParameter(const String&, const String&, Mode);

    typedef HashMap<String, String> KeyValuePairs;
    String m_contentType;
    KeyValuePairs m_parameterValues;
    Vector<String> m_parameterNames;
    String m_mimeType;
};

}
