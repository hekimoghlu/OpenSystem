/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "HeapSnapshot.h"

#include <wtf/DataLog.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HeapSnapshot);

HeapSnapshot::HeapSnapshot(HeapSnapshot* previousSnapshot)
    : m_previous(previousSnapshot)
{
}

HeapSnapshot::~HeapSnapshot() = default;

void HeapSnapshot::appendNode(const HeapSnapshotNode& node)
{
    ASSERT(!m_finalized);
    ASSERT(!m_previous || !m_previous->nodeForCell(node.cell));

    m_nodes.append(node);
    m_filter.add(std::bit_cast<uintptr_t>(node.cell));
}

void HeapSnapshot::sweepCell(JSCell* cell)
{
    ASSERT(cell);

    if (m_finalized && !m_filter.ruleOut(std::bit_cast<uintptr_t>(cell))) {
        ASSERT_WITH_MESSAGE(!isEmpty(), "Our filter should have ruled us out if we are empty.");
        unsigned start = 0;
        unsigned end = m_nodes.size();
        while (start != end) {
            unsigned middle = start + ((end - start) / 2);
            HeapSnapshotNode& node = m_nodes[middle];
            if (cell == node.cell) {
                // Cells should always have 0 as low bits.
                // Mark this cell for removal by setting the low bit.
                ASSERT(!(reinterpret_cast<intptr_t>(node.cell) & CellToSweepTag));
                node.cell = reinterpret_cast<JSCell*>(reinterpret_cast<intptr_t>(node.cell) | CellToSweepTag);
                m_hasCellsToSweep = true;
                return;
            }
            if (cell < node.cell)
                end = middle;
            else
                start = middle + 1;
        }
    }

    if (m_previous)
        m_previous->sweepCell(cell);
}

void HeapSnapshot::shrinkToFit()
{
    if (m_finalized && m_hasCellsToSweep) {
        m_filter.reset();
        m_nodes.removeAllMatching(
            [&] (const HeapSnapshotNode& node) -> bool {
                bool willRemoveCell = std::bit_cast<intptr_t>(node.cell) & CellToSweepTag;
                if (!willRemoveCell)
                    m_filter.add(std::bit_cast<uintptr_t>(node.cell));
                return willRemoveCell;
            });
        m_nodes.shrinkToFit();
        m_hasCellsToSweep = false;
    }

    if (m_previous)
        m_previous->shrinkToFit();
}

void HeapSnapshot::finalize()
{
    ASSERT(!m_finalized);
    m_finalized = true;

    // Nodes are appended to the snapshot in identifier order.
    // Now that we have the complete list of nodes we will sort
    // them in a different order. Remember the range of identifiers
    // in this snapshot.
    if (!isEmpty()) {
        m_firstObjectIdentifier = m_nodes.first().identifier;
        m_lastObjectIdentifier = m_nodes.last().identifier;
    }

    std::sort(m_nodes.begin(), m_nodes.end(), [] (const HeapSnapshotNode& a, const HeapSnapshotNode& b) {
        return a.cell < b.cell;
    });

#ifndef NDEBUG
    // Assert there are no duplicates or nullptr cells.
    JSCell* previousCell = nullptr;
    for (auto& node : m_nodes) {
        ASSERT(node.cell);
        ASSERT(!(reinterpret_cast<intptr_t>(node.cell) & CellToSweepTag));
        if (node.cell == previousCell) {
            dataLog("Seeing same cell twice: ", RawPointer(previousCell), "\n");
            ASSERT(node.cell != previousCell);
        }
        previousCell = node.cell;
    }
#endif
}

std::optional<HeapSnapshotNode> HeapSnapshot::nodeForCell(JSCell* cell)
{
    ASSERT(m_finalized);

    if (!m_filter.ruleOut(std::bit_cast<uintptr_t>(cell))) {
        ASSERT_WITH_MESSAGE(!isEmpty(), "Our filter should have ruled us out if we are empty.");
        unsigned start = 0;
        unsigned end = m_nodes.size();
        while (start != end) {
            unsigned middle = start + ((end - start) / 2);
            HeapSnapshotNode& node = m_nodes[middle];
            if (cell == node.cell)
                return std::optional<HeapSnapshotNode>(node);
            if (cell < node.cell)
                end = middle;
            else
                start = middle + 1;
        }
    }

    if (m_previous)
        return m_previous->nodeForCell(cell);

    return std::nullopt;
}

std::optional<HeapSnapshotNode> HeapSnapshot::nodeForObjectIdentifier(unsigned objectIdentifier)
{
    if (isEmpty()) {
        if (m_previous)
            return m_previous->nodeForObjectIdentifier(objectIdentifier);
        return std::nullopt;
    }

    if (objectIdentifier > m_lastObjectIdentifier)
        return std::nullopt;

    if (objectIdentifier < m_firstObjectIdentifier) {
        if (m_previous)
            return m_previous->nodeForObjectIdentifier(objectIdentifier);
        return std::nullopt;
    }

    for (auto& node : m_nodes) {
        if (node.identifier == objectIdentifier)
            return std::optional<HeapSnapshotNode>(node);
    }

    return std::nullopt;
}

} // namespace JSC
