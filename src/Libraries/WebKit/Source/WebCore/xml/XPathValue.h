/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

#include "XPathNodeSet.h"

namespace WebCore {
namespace XPath {

class Value {
public:
    enum class Type : uint8_t { NodeSet, Boolean, Number, String };

    Value() = delete;

    Value(bool value)
        : m_type(Type::Boolean), m_bool(value)
    { }
    Value(unsigned value)
        : m_type(Type::Number), m_number(value)
    { }
    Value(double value)
        : m_type(Type::Number), m_number(value)
    { }

    Value(const String& value)
        : m_type(Type::String), m_data(Data::create(value))
    { }
    Value(const char* value)
        : m_type(Type::String), m_data(Data::create(String::fromLatin1(value)))
    { }

    explicit Value(NodeSet&& value)
        : m_type(Type::NodeSet), m_data(Data::create(WTFMove(value)))
    { }
    explicit Value(Node* value)
        : m_type(Type::NodeSet), m_data(Data::create(value))
    { }
    explicit Value(RefPtr<Node>&& value)
        : m_type(Type::NodeSet), m_data(Data::create(WTFMove(value)))
    { }

    Type type() const { return m_type; }

    bool isNodeSet() const { return m_type == Type::NodeSet; }
    bool isBoolean() const { return m_type == Type::Boolean; }
    bool isNumber() const { return m_type == Type::Number; }
    bool isString() const { return m_type == Type::String; }

    const NodeSet& toNodeSet() const;
    bool toBoolean() const;
    double toNumber() const;
    String toString() const;

    // Note that the NodeSet is shared with other Values that this one was copied from or that are copies of this one.
    NodeSet& modifiableNodeSet();

private:
    // This constructor creates ambiguity so that we don't accidentally call the boolean overload for pointer types.
    Value(void*) = delete;

    struct Data : public RefCounted<Data> {
        static Ref<Data> create() { return adoptRef(*new Data); }
        static Ref<Data> create(const String& string) { return adoptRef(*new Data(string)); }
        static Ref<Data> create(NodeSet&& nodeSet) { return adoptRef(*new Data(WTFMove(nodeSet))); }
        static Ref<Data> create(RefPtr<Node>&& node) { return adoptRef(*new Data(WTFMove(node))); }

        String string;
        NodeSet nodeSet;

    private:
        Data() { }
        explicit Data(const String& string)
            : string(string)
        { }
        explicit Data(NodeSet&& nodeSet)
            : nodeSet(WTFMove(nodeSet))
        { }
        explicit Data(RefPtr<Node>&& node)
            : nodeSet(WTFMove(node))
        { }
    };

    Type m_type;
    bool m_bool { false };
    double m_number { 0 };
    RefPtr<Data> m_data;
};

} // namespace XPath
} // namespace WebCore
