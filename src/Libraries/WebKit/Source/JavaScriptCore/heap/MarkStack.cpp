/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include "MarkStack.h"

#include "GCSegmentedArrayInlines.h"

namespace JSC {

MarkStackArray::MarkStackArray()
    : GCSegmentedArray<const JSCell*>()
{
}

void MarkStackArray::transferTo(MarkStackArray& other)
{
    RELEASE_ASSERT(this != &other);
    
    // Remove our head and the head of the other list.
    GCArraySegment<const JSCell*>* myHead = m_segments.removeHead();
    GCArraySegment<const JSCell*>* otherHead = other.m_segments.removeHead();
    m_numberOfSegments--;
    other.m_numberOfSegments--;
    
    other.m_segments.append(m_segments);
    
    other.m_numberOfSegments += m_numberOfSegments;
    m_numberOfSegments = 0;
    
    // Put the original heads back in their places.
    m_segments.push(myHead);
    other.m_segments.push(otherHead);
    m_numberOfSegments++;
    other.m_numberOfSegments++;
    
    while (!isEmpty()) {
        refill();
        while (canRemoveLast())
            other.append(removeLast());
    }
}

size_t MarkStackArray::transferTo(MarkStackArray& other, size_t limit)
{
    size_t count = 0;
    while (count < limit && !isEmpty()) {
        refill();
        while (count < limit && canRemoveLast()) {
            other.append(removeLast());
            count++;
        }
    }
    RELEASE_ASSERT(count <= limit);
    return count;
}

void MarkStackArray::donateSomeCellsTo(MarkStackArray& other)
{
    // Try to donate about 1 / 2 of our cells. To reduce copying costs,
    // we prefer donating whole segments over donating individual cells,
    // even if this skews away from our 1 / 2 target.

    size_t segmentsToDonate = m_numberOfSegments / 2; // If we only have one segment (our head) we don't donate any segments.

    if (!segmentsToDonate) {
        size_t cellsToDonate = m_top / 2; // Round down to donate 0 / 1 cells.
        while (cellsToDonate--) {
            ASSERT(m_top);
            other.append(removeLast());
        }
        return;
    }

    validatePrevious();
    other.validatePrevious();

    // Remove our head and the head of the other list before we start moving segments around.
    // We'll add them back on once we're done donating.
    GCArraySegment<const JSCell*>* myHead = m_segments.removeHead();
    GCArraySegment<const JSCell*>* otherHead = other.m_segments.removeHead();

    while (segmentsToDonate--) {
        GCArraySegment<const JSCell*>* current = m_segments.removeHead();
        ASSERT(current);
        ASSERT(m_numberOfSegments > 1);
        other.m_segments.push(current);
        m_numberOfSegments--;
        other.m_numberOfSegments++;
    }

    // Put the original heads back in their places.
    m_segments.push(myHead);
    other.m_segments.push(otherHead);

    validatePrevious();
    other.validatePrevious();
}

void MarkStackArray::stealSomeCellsFrom(MarkStackArray& other, size_t idleThreadCount)
{
    // Try to steal 1 / Nth of the shared array, where N is the number of idle threads.
    // To reduce copying costs, we prefer stealing a whole segment over stealing
    // individual cells, even if this skews away from our 1 / N target.

    validatePrevious();
    other.validatePrevious();
        
    // If other has an entire segment, steal it and return.
    if (other.m_numberOfSegments > 1) {
        // Move the heads of the lists aside. We'll push them back on after.
        GCArraySegment<const JSCell*>* otherHead = other.m_segments.removeHead();
        GCArraySegment<const JSCell*>* myHead = m_segments.removeHead();

        ASSERT(other.m_segments.head()->m_top == s_segmentCapacity);

        m_segments.push(other.m_segments.removeHead());

        m_numberOfSegments++;
        other.m_numberOfSegments--;
        
        m_segments.push(myHead);
        other.m_segments.push(otherHead);
    
        validatePrevious();
        other.validatePrevious();
        return;
    }

    // Steal ceil(other.size() / idleThreadCount) things.
    size_t numberOfCellsToSteal = (other.size() + idleThreadCount - 1) / idleThreadCount;
    while (numberOfCellsToSteal-- > 0 && other.canRemoveLast())
        append(other.removeLast());
}

} // namespace JSC
