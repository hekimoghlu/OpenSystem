/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

#include <wtf/text/WTFString.h>

namespace WTF {
class URL;
}

namespace WebCore {

class ContentType {
public:
    WEBCORE_EXPORT explicit ContentType(String&& type);
    WEBCORE_EXPORT explicit ContentType(const String& type);
    WEBCORE_EXPORT ContentType(const String& type, bool); // Use by the IPC serializer.
    ContentType() = default;

    WEBCORE_EXPORT static ContentType fromURL(const WTF::URL&);

    WEBCORE_EXPORT static const String& codecsParameter();
    static const String& profilesParameter();

    WEBCORE_EXPORT String parameter(const String& parameterName) const;
    WEBCORE_EXPORT String containerType() const;
    Vector<String> codecs() const;
    Vector<String> profiles() const;
    const String& raw() const { return m_type; }
    bool isEmpty() const { return m_type.isEmpty(); }

    bool typeWasInferredFromExtension() const { return m_typeWasInferredFromExtension; }

    WEBCORE_EXPORT String toJSONString() const;
    bool operator==(const ContentType& other) const { return raw() == other.raw(); }
    bool operator!=(const ContentType& other) const { return !(*this == other); }

    ContentType isolatedCopy() const & { return { m_type.isolatedCopy(), m_typeWasInferredFromExtension }; }
    ContentType isolatedCopy() && { return { WTFMove(m_type).isolatedCopy(), m_typeWasInferredFromExtension }; }

private:
    String m_type;
    bool m_typeWasInferredFromExtension { false };
};

} // namespace WebCore

namespace WTF {
template<typename Type> struct LogArgument;

template <>
struct LogArgument<WebCore::ContentType> {
    static String toString(const WebCore::ContentType& type)
    {
        return type.toJSONString();
    }
};

} // namespace WTF
