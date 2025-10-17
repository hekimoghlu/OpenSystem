/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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

#include <wtf/CommaPrinter.h>
#include <wtf/PrintStream.h>
#include <wtf/StringPrintStream.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

template<typename T>
class ListDump {
public:
    ListDump(const T& list, ASCIILiteral comma)
        : m_list(list)
        , m_comma(comma)
    {
    }
    
    void dump(PrintStream& out) const
    {
        for (auto iter = m_list.begin(); iter != m_list.end(); ++iter)
            out.print(m_comma, *iter);
    }

private:
    const T& m_list;
    CommaPrinter m_comma;
};

template<typename T>
class PointerListDump {
public:
    PointerListDump(const T& list, ASCIILiteral comma)
        : m_list(list)
        , m_comma(comma)
    {
    }
    
    void dump(PrintStream& out) const
    {
        for (auto iter = m_list.begin(); iter != m_list.end(); ++iter)
            out.print(m_comma, pointerDump(*iter));
    }

private:
    const T& m_list;
    CommaPrinter m_comma;
};

template<typename T>
class MapDump {
public:
    MapDump(const T& map, ASCIILiteral arrow, ASCIILiteral comma)
        : m_map(map)
        , m_arrow(arrow)
        , m_comma(comma)
    {
    }
    
    void dump(PrintStream& out) const
    {
        for (auto iter = m_map.begin(); iter != m_map.end(); ++iter)
            out.print(m_comma, iter->key, m_arrow, iter->value);
    }
    
private:
    const T& m_map;
    ASCIILiteral m_arrow;
    CommaPrinter m_comma;
};

template<typename T>
ListDump<T> listDump(const T& list, ASCIILiteral comma = ", "_s)
{
    return ListDump<T>(list, comma);
}

template<typename T>
PointerListDump<T> pointerListDump(const T& list, ASCIILiteral comma = ", "_s)
{
    return PointerListDump<T>(list, comma);
}

template<typename T, typename Comparator>
CString sortedListDump(const T& list, const Comparator& comparator, ASCIILiteral comma = ", "_s)
{
    Vector<typename T::ValueType> myList;
    myList.appendRange(list.begin(), list.end());
    std::sort(myList.begin(), myList.end(), comparator);
    StringPrintStream out;
    CommaPrinter commaPrinter(comma);
    for (unsigned i = 0; i < myList.size(); ++i)
        out.print(commaPrinter, myList[i]);
    return out.toCString();
}

template<typename T>
CString sortedListDump(const T& list, ASCIILiteral comma = ", "_s)
{
    return sortedListDump(list, std::less<>(), comma);
}

template<typename T>
MapDump<T> mapDump(const T& map, ASCIILiteral arrow = "=>"_s, ASCIILiteral comma = ", "_s)
{
    return MapDump<T>(map, arrow, comma);
}

template<typename T, typename Comparator>
CString sortedMapDump(const T& map, const Comparator& comparator, ASCIILiteral arrow = "=>"_s, ASCIILiteral comma = ", "_s)
{
    Vector<typename T::KeyType> keys;
    for (auto iter = map.begin(); iter != map.end(); ++iter)
        keys.append(iter->key);
    std::sort(keys.begin(), keys.end(), comparator);
    StringPrintStream out;
    CommaPrinter commaPrinter(comma);
    for (unsigned i = 0; i < keys.size(); ++i)
        out.print(commaPrinter, keys[i], arrow, map.get(keys[i]));
    return out.toCString();
}

template<typename T, typename U>
class ListDumpInContext {
public:
    ListDumpInContext(const T& list, U* context, ASCIILiteral comma)
        : m_list(list)
        , m_context(context)
        , m_comma(comma)
    {
    }
    
    void dump(PrintStream& out) const
    {
        for (auto iter = m_list.begin(); iter != m_list.end(); ++iter)
            out.print(m_comma, inContext(*iter, m_context));
    }

private:
    const T& m_list;
    U* m_context;
    CommaPrinter m_comma;
};

template<typename T, typename U>
ListDumpInContext<T, U> listDumpInContext(
    const T& list, U* context, ASCIILiteral comma = ", "_s)
{
    return ListDumpInContext<T, U>(list, context, comma);
}

} // namespace WTF

using WTF::listDump;
using WTF::listDumpInContext;
using WTF::mapDump;
using WTF::pointerListDump;
using WTF::sortedListDump;
using WTF::sortedMapDump;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
