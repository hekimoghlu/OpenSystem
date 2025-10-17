/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include "CallGraph.h"

#include "AST.h"
#include "ASTVisitor.h"
#include "WGSLShaderModule.h"
#include <wtf/Deque.h>

namespace WGSL {

class CallGraphBuilder : public AST::Visitor {
public:
    CallGraphBuilder(ShaderModule& shaderModule)
        : m_shaderModule(shaderModule)
    {
    }

    void build();

    void visit(AST::Function&) override;
    void visit(AST::CallExpression&) override;

private:
    void initializeMappings();

    ShaderModule& m_shaderModule;
    CallGraph m_callGraph;
    HashMap<AST::Function*, unsigned> m_calleeBuildingMap;
    Vector<CallGraph::Callee>* m_callees { nullptr };
    AST::Function* m_currentFunction { nullptr };
    Deque<AST::Function*> m_queue;
};

void CallGraphBuilder::build()
{
    initializeMappings();
    m_shaderModule.setCallGraph(WTFMove(m_callGraph));
}

void CallGraphBuilder::initializeMappings()
{
    for (auto& declaration : m_shaderModule.declarations()) {
        auto* function = dynamicDowncast<AST::Function>(declaration);
        if (!function)
            continue;

        const auto& name = function->name();
        {
            auto result = m_callGraph.m_functionsByName.add(name, function);
            ASSERT_UNUSED(result, result.isNewEntry);
        }

        if (!function->stage())
            continue;

        m_callGraph.m_entrypoints.append({ *function, *function->stage(), function->name() });
        m_queue.append(function);
    }

    while (!m_queue.isEmpty())
        visit(*m_queue.takeFirst());
}

void CallGraphBuilder::visit(AST::Function& function)
{
    auto result = m_callGraph.m_calleeMap.add(&function, Vector<CallGraph::Callee>());
    if (!result.isNewEntry)
        return;

    ASSERT(!m_callees);
    m_callees = &result.iterator->value;
    m_currentFunction = &function;
    AST::Visitor::visit(function);
    m_calleeBuildingMap.clear();
    m_currentFunction = nullptr;
    m_callees = nullptr;
}

void CallGraphBuilder::visit(AST::CallExpression& call)
{
    for (auto& argument : call.arguments())
        AST::Visitor::visit(argument);

    auto* target = dynamicDowncast<AST::IdentifierExpression>(call.target());
    if (!target)
        return;

    auto it = m_callGraph.m_functionsByName.find(target->identifier());
    if (it == m_callGraph.m_functionsByName.end())
        return;

    m_queue.append(it->value);
    auto result = m_calleeBuildingMap.add(it->value, m_callees->size());
    if (result.isNewEntry)
        m_callees->append(CallGraph::Callee { it->value, { { m_currentFunction, &call } } });
    else
        m_callees->at(result.iterator->value).callSites.append({ m_currentFunction, &call });
}

void buildCallGraph(ShaderModule& shaderModule)
{
    CallGraphBuilder(shaderModule).build();
}

} // namespace WGSL
