/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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

#include "ASTForward.h"
#include "WGSLEnums.h"
#include <wtf/HashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

class ShaderModule;
struct PipelineLayout;
struct PrepareResult;

class CallGraph {
    friend class CallGraphBuilder;

public:
    struct Callee {
        AST::Function* target;
        Vector<std::tuple<AST::Function*, AST::CallExpression*>> callSites;
    };

    struct EntryPoint {
        AST::Function& function;
        ShaderStage stage;
        String originalName;
    };

    const Vector<EntryPoint>& entrypoints() const { return m_entrypoints; }
    const Vector<Callee>& callees(AST::Function& function) const { return m_calleeMap.find(&function)->value; }

private:
    CallGraph() { }

    Vector<EntryPoint> m_entrypoints;
    HashMap<String, AST::Function*> m_functionsByName;
    HashMap<AST::Function*, Vector<Callee>> m_calleeMap;
};

void buildCallGraph(ShaderModule&);

} // namespace WGSL
