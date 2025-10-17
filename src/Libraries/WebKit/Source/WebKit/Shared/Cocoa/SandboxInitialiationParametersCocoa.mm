/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
#import "config.h"
#import "SandboxInitializationParameters.h"

namespace WebKit {

SandboxInitializationParameters::SandboxInitializationParameters()
    : m_profileSelectionMode(ProfileSelectionMode::UseDefaultSandboxProfilePath)
{
}

SandboxInitializationParameters::~SandboxInitializationParameters() = default;

void SandboxInitializationParameters::appendPathInternal(ASCIILiteral name, const char* path)
{
    std::array<char, PATH_MAX> normalizedPath;
    if (!realpath(path, normalizedPath.data()))
        normalizedPath[0] = '\0';

    m_parameterNames.append(name);
    m_parameterValues.append(normalizedPath.data());
}

void SandboxInitializationParameters::addConfDirectoryParameter(ASCIILiteral name, int confID)
{
    std::array<char, PATH_MAX> path;
    if (confstr(confID, path.data(), PATH_MAX) <= 0)
        path[0] = '\0';

    appendPathInternal(name, path.data());
}

void SandboxInitializationParameters::addPathParameter(ASCIILiteral name, NSString *path)
{
    appendPathInternal(name, [path length] ? [(NSString *)path fileSystemRepresentation] : "");
}

void SandboxInitializationParameters::addPathParameter(ASCIILiteral name, const char* path)
{
    appendPathInternal(name, path);
}

void SandboxInitializationParameters::addParameter(ASCIILiteral name, CString&& value)
{
    m_parameterNames.append(name);
    m_parameterValues.append(WTFMove(value));
}

Vector<const char*> SandboxInitializationParameters::namedParameterVector() const
{
    Vector<const char*> result;
    result.reserveInitialCapacity(m_parameterNames.size() * 2 + 1);
    ASSERT(m_parameterNames.size() == m_parameterValues.size());
    for (size_t i = 0; i < m_parameterNames.size(); ++i) {
        result.append(m_parameterNames[i]);
        result.append(m_parameterValues[i].data());
    }
    result.append(nullptr);
    return result;
}

size_t SandboxInitializationParameters::count() const
{
    return m_parameterNames.size();
}

ASCIILiteral SandboxInitializationParameters::name(size_t index) const
{
    return m_parameterNames[index];
}

const char* SandboxInitializationParameters::value(size_t index) const
{
    return m_parameterValues[index].data();
}

} // namespace WebKit
