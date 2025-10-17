/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "SerializedNFA.h"

#include "NFA.h"
#include <wtf/text/ParsingUtilities.h>

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore {
namespace ContentExtensions {

template<typename T>
bool writeAllToFile(FileSystem::PlatformFileHandle file, const T& container)
{
    auto bytes = spanReinterpretCast<const uint8_t>(container.span());
    while (!bytes.empty()) {
        auto written = FileSystem::writeToFile(file, bytes);
        if (written == -1)
            return false;
        skip(bytes, written);
    }
    return true;
}

std::optional<SerializedNFA> SerializedNFA::serialize(NFA&& nfa)
{
    auto [filename, file] = FileSystem::openTemporaryFile("SerializedNFA"_s);
    if (!FileSystem::isHandleValid(file))
        return std::nullopt;

    bool wroteSuccessfully = writeAllToFile(file, nfa.nodes)
        && writeAllToFile(file, nfa.transitions)
        && writeAllToFile(file, nfa.targets)
        && writeAllToFile(file, nfa.epsilonTransitionsTargets)
        && writeAllToFile(file, nfa.actions);
    if (!wroteSuccessfully) {
        FileSystem::closeFile(file);
        FileSystem::deleteFile(filename);
        return std::nullopt;
    }

    bool mappedSuccessfully = false;
    FileSystem::MappedFileData mappedFile(file, FileSystem::MappedFileMode::Private, mappedSuccessfully);
    FileSystem::closeFile(file);
    FileSystem::deleteFile(filename);
    if (!mappedSuccessfully)
        return std::nullopt;

    Metadata metadata {
        nfa.nodes.size(),
        nfa.transitions.size(),
        nfa.targets.size(),
        nfa.epsilonTransitionsTargets.size(),
        nfa.actions.size(),
        0,
        nfa.nodes.size() * sizeof(nfa.nodes[0]),
        nfa.nodes.size() * sizeof(nfa.nodes[0])
            + nfa.transitions.size() * sizeof(nfa.transitions[0]),
        nfa.nodes.size() * sizeof(nfa.nodes[0])
            + nfa.transitions.size() * sizeof(nfa.transitions[0])
            + nfa.targets.size() * sizeof(nfa.targets[0]),
        nfa.nodes.size() * sizeof(nfa.nodes[0])
            + nfa.transitions.size() * sizeof(nfa.transitions[0])
            + nfa.targets.size() * sizeof(nfa.targets[0])
            + nfa.epsilonTransitionsTargets.size() * sizeof(nfa.epsilonTransitionsTargets[0])
    };

    nfa.clear();

    return {{ WTFMove(mappedFile), WTFMove(metadata) }};
}

SerializedNFA::SerializedNFA(FileSystem::MappedFileData&& file, Metadata&& metadata)
    : m_file(WTFMove(file))
    , m_metadata(WTFMove(metadata))
{
}

template<typename T>
std::span<const T> SerializedNFA::spanAtOffsetInFile(size_t offset, size_t length) const
{
    return spanReinterpretCast<const T>(m_file.span().subspan(offset).first(length * sizeof(T)));
}

auto SerializedNFA::nodes() const -> const Range<ImmutableNFANode>
{
    return spanAtOffsetInFile<ImmutableNFANode>(m_metadata.nodesOffset, m_metadata.nodesSize);
}

auto SerializedNFA::transitions() const -> const Range<ImmutableRange<char>>
{
    return spanAtOffsetInFile<ImmutableRange<char>>(m_metadata.transitionsOffset, m_metadata.transitionsSize);
}

auto SerializedNFA::targets() const -> const Range<uint32_t>
{
    return spanAtOffsetInFile<uint32_t>(m_metadata.targetsOffset, m_metadata.targetsSize);
}

auto SerializedNFA::epsilonTransitionsTargets() const -> const Range<uint32_t>
{
    return spanAtOffsetInFile<uint32_t>(m_metadata.epsilonTransitionsTargetsOffset, m_metadata.epsilonTransitionsTargetsSize);
}

auto SerializedNFA::actions() const -> const Range<uint64_t>
{
    return spanAtOffsetInFile<uint64_t>(m_metadata.actionsOffset, m_metadata.actionsSize);
}

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
