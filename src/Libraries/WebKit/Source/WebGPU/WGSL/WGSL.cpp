/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#include "WGSL.h"

#include "ASTIdentifierExpression.h"
#include "AttributeValidator.h"
#include "BoundsCheck.h"
#include "CallGraph.h"
#include "EntryPointRewriter.h"
#include "GlobalSorting.h"
#include "GlobalVariableRewriter.h"
#include "MangleNames.h"
#include "Metal/MetalCodeGenerator.h"
#include "Parser.h"
#include "PhaseTimer.h"
#include "PointerRewriter.h"
#include "TypeCheck.h"
#include "VisibilityValidator.h"
#include "WGSLShaderModule.h"

namespace WGSL {

#define CHECK_PASS(pass, ...) \
    dumpASTBetweenEachPassIfNeeded(shaderModule, "AST before " # pass); \
    auto maybe##pass##Failure = [&]() { \
        PhaseTimer phaseTimer(#pass, phaseTimes); \
        return pass(__VA_ARGS__); \
    }(); \
    if (maybe##pass##Failure) \
        return { *maybe##pass##Failure };

#define RUN_PASS(pass, ...) \
    do { \
        PhaseTimer phaseTimer(#pass, phaseTimes); \
        dumpASTBetweenEachPassIfNeeded(shaderModule, "AST before " # pass); \
        pass(__VA_ARGS__); \
    } while (0)

#define RUN_PASS_WITH_RESULT(name, pass, ...) \
    dumpASTBetweenEachPassIfNeeded(shaderModule, "AST before " # pass); \
    auto name = [&]() { \
        PhaseTimer phaseTimer(#pass, phaseTimes); \
        return pass(__VA_ARGS__); \
    }();

std::variant<SuccessfulCheck, FailedCheck> staticCheck(const String& wgsl, const std::optional<SourceMap>&, const Configuration& configuration)
{
    PhaseTimes phaseTimes;
    auto shaderModule = makeUniqueRef<ShaderModule>(wgsl, configuration);

    CHECK_PASS(parse, shaderModule);
    CHECK_PASS(reorderGlobals, shaderModule);
    CHECK_PASS(typeCheck, shaderModule);
    CHECK_PASS(validateAttributes, shaderModule);
    RUN_PASS(buildCallGraph, shaderModule);
    CHECK_PASS(validateIO, shaderModule);
    CHECK_PASS(validateVisibility, shaderModule);
    RUN_PASS(mangleNames, shaderModule);

    Vector<Warning> warnings { };
    return std::variant<SuccessfulCheck, FailedCheck>(std::in_place_type<SuccessfulCheck>, WTFMove(warnings), WTFMove(shaderModule));
}

SuccessfulCheck::SuccessfulCheck(SuccessfulCheck&&) = default;

SuccessfulCheck::SuccessfulCheck(Vector<Warning>&& messages, UniqueRef<ShaderModule>&& shader)
    : warnings(WTFMove(messages))
    , ast(WTFMove(shader))
{
}

SuccessfulCheck::~SuccessfulCheck() = default;

inline std::variant<PrepareResult, Error> prepareImpl(ShaderModule& shaderModule, const HashMap<String, PipelineLayout*>& pipelineLayouts)
{
    CompilationScope compilationScope(shaderModule);

    PhaseTimes phaseTimes;
    auto result = [&]() -> std::variant<PrepareResult, Error> {
        PhaseTimer phaseTimer("prepare total", phaseTimes);

        HashMap<String, Reflection::EntryPointInformation> entryPoints;

        RUN_PASS(insertBoundsChecks, shaderModule);
        RUN_PASS(rewritePointers, shaderModule);
        RUN_PASS(rewriteEntryPoints, shaderModule, pipelineLayouts);
        CHECK_PASS(rewriteGlobalVariables, shaderModule, pipelineLayouts, entryPoints);

        dumpASTAtEndIfNeeded(shaderModule);

        return { PrepareResult { WTFMove(entryPoints), WTFMove(compilationScope) } };
    }();

    logPhaseTimes(phaseTimes);

    return result;
}

std::variant<String, Error> generate(ShaderModule& shaderModule, PrepareResult& prepareResult, HashMap<String, ConstantValue>& constantValues)
{
    PhaseTimes phaseTimes;
    String result;
    if (auto maybeError = shaderModule.validateOverrides(constantValues))
        return { *maybeError };
    {
        PhaseTimer phaseTimer("generateMetalCode", phaseTimes);
        result = Metal::generateMetalCode(shaderModule, prepareResult, constantValues);
    }
    logPhaseTimes(phaseTimes);
    return { result };
}

std::variant<PrepareResult, Error> prepare(ShaderModule& ast, const HashMap<String, PipelineLayout*>& pipelineLayouts)
{
    return prepareImpl(ast, pipelineLayouts);
}

std::variant<PrepareResult, Error> prepare(ShaderModule& ast, const String& entryPointName, PipelineLayout* pipelineLayout)
{
    HashMap<String, PipelineLayout*> pipelineLayouts;
    pipelineLayouts.add(entryPointName, pipelineLayout);
    return prepareImpl(ast, pipelineLayouts);
}

std::optional<ConstantValue> evaluate(const AST::Expression& expression, const HashMap<String, ConstantValue>& constants)
{
    if (auto constantValue = expression.constantValue())
        return *constantValue;
    auto* maybeIdentifierExpression = dynamicDowncast<const AST::IdentifierExpression>(expression);
    if (!maybeIdentifierExpression)
        return std::nullopt;
    auto it = constants.find(maybeIdentifierExpression->identifier());
    if (it == constants.end())
        return std::nullopt;
    const_cast<AST::Expression&>(expression).setConstantValue(it->value);
    return it->value;
}

}
