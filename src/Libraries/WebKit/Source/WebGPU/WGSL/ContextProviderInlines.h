/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include "ASTIdentifier.h"
#include "ContextProvider.h"

namespace WGSL {

template<typename Value>
ContextProvider<Value>::Context::Context(const Context *const parent)
    : m_parent(parent)
{
}

template<typename Value>
const Value* ContextProvider<Value>::Context::lookup(const String& name) const
{
    auto it = m_map.find(name);
    if (it != m_map.end())
        return &it->value;
    if (m_parent)
        return m_parent->lookup(name);
    return nullptr;
}

template<typename Value>
const Value* ContextProvider<Value>::Context::add(const String& name, const Value& value)
{
    auto result = m_map.add(name, value);
    if (UNLIKELY(!result.isNewEntry))
        return nullptr;
    return &result.iterator->value;
}

template<typename Value>
ContextProvider<Value>::ContextScope::ContextScope(ContextProvider<Value>* provider)
    : m_provider(*provider)
    , m_previousContext(provider->m_context)
{
    m_provider.m_contexts.append(std::unique_ptr<Context>(new Context { m_previousContext }));
    m_provider.m_context = m_provider.m_contexts.last().get();
}

template<typename Value>
ContextProvider<Value>::ContextScope::~ContextScope()
{
    m_provider.m_context = m_previousContext;
    m_provider.m_contexts.removeLast();
}

template<typename Value>
ContextProvider<Value>::ContextProvider()
    : m_context(nullptr)
    , m_contexts()
    , m_globalScope(this)
{
}

template<typename Value>
auto ContextProvider<Value>::introduceVariable(const String& name, const Value& value) -> const Value*
{
    return m_context->add(name, value);
}

template<typename Value>
auto ContextProvider<Value>::readVariable(const String& name) const -> const Value*
{
    return m_context->lookup(name);
}

} // namespace WGSL
