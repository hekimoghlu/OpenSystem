/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#include "CompilationScope.h"

#include "WGSLShaderModule.h"

namespace WGSL {

CompilationScope::CompilationScope(ShaderModule& shaderModule)
    : m_shaderModule(shaderModule)
    , m_builderState(shaderModule.astBuilder().saveCurrentState())
    , m_replacementsSize(shaderModule.currentReplacementSize())
{
}

CompilationScope::CompilationScope(CompilationScope&& other)
    : m_shaderModule(other.m_shaderModule)
    , m_builderState(other.m_builderState)
    , m_replacementsSize(other.m_replacementsSize)
{
    other.m_invalidated = true;
}

CompilationScope::~CompilationScope()
{
    if (m_invalidated)
        return;
    m_shaderModule.revertReplacements(m_replacementsSize);
    m_shaderModule.astBuilder().restore(WTFMove(m_builderState));
}

} // namespace WGSL
