/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#if ENABLE(VIDEO)

namespace WebCore {

class WebVTTTokenTypes {
public:
    enum Type {
        Uninitialized,
        Character,
        StartTag,
        EndTag,
        TimestampTag,
    };
};

class WebVTTToken {
public:
    typedef WebVTTTokenTypes Type;

    WebVTTToken()
        : m_type(Type::Uninitialized) { }

    static WebVTTToken StringToken(const String& characterData)
    {
        return WebVTTToken(Type::Character, characterData);
    }

    static WebVTTToken StartTag(const String& tagName, const AtomString& classes = emptyAtom(), const AtomString& annotation = emptyAtom())
    {
        WebVTTToken token(Type::StartTag, tagName);
        token.m_classes = classes;
        token.m_annotation = annotation;
        return token;
    }

    static WebVTTToken EndTag(const String& tagName)
    {
        return WebVTTToken(Type::EndTag, tagName);
    }

    static WebVTTToken TimestampTag(const String& timestampData)
    {
        return WebVTTToken(Type::TimestampTag, timestampData);
    }

    Type::Type type() const { return m_type; }
    const String& name() const { return m_data; }
    const String& characters() const { return m_data; }
    const AtomString& classes() const { return m_classes; }
    const AtomString& annotation() const { return m_annotation; }

private:
    WebVTTToken(Type::Type type, const String& data)
        : m_type(type)
        , m_data(data) { }

    Type::Type m_type;
    String m_data;
    AtomString m_annotation;
    AtomString m_classes;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
