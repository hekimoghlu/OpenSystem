/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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

#include "ASTNode.h"
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/PrintStream.h>
#include <wtf/text/WTFString.h>

namespace WGSL::AST {

class Identifier : public Node {
    WTF_FORBID_HEAP_ALLOCATION;
public:
    static Identifier make(const String& id) { return { SourceSpan::empty(), String(id) }; }
    static Identifier make(String&& id) { return { SourceSpan::empty(), WTFMove(id) }; }
    static Identifier makeWithSpan(SourceSpan span, String&& id) { return { WTFMove(span), WTFMove(id) }; }
    static Identifier makeWithSpan(SourceSpan span, StringView id) { return { WTFMove(span), id }; }

    NodeKind kind() const override;
    operator String&() { return m_id; }
    operator const String&() const { return m_id; }

    const String& id() const { return m_id; }

    void dump(PrintStream&) const;

private:
    Identifier(const SourceSpan& span, String&& id)
        : Node(span)
        , m_id(WTFMove(id))
    { }

    Identifier(const SourceSpan& span, StringView id)
        : Identifier(span, id.toString())
    { }

    String m_id;
};

inline void Identifier::dump(PrintStream& out) const
{
    out.print(m_id);
}

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_WGSL_AST(Identifier);

namespace WTF {

template<> class StringTypeAdapter<WGSL::AST::Identifier, void> : public StringTypeAdapter<StringImpl*, void> {
public:
    StringTypeAdapter(const WGSL::AST::Identifier& id)
        : StringTypeAdapter<StringImpl*, void> { id.id().impl() }
    { }
};

} // namespace WTF
