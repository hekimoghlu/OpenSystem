/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

#include "PODInterval.h"
#include "PODRedBlackTree.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

// FIXME: The prefix "POD" here isn't correct; this tree works with non-POD types.

namespace WebCore {

struct PODIntervalNodeUpdater;

// An interval tree, which is a form of augmented red-black tree. It
// supports efficient (O(lg n)) insertion, removal and querying of
// intervals in the tree.
template<typename T, typename UserData> class PODIntervalTree final : public PODRedBlackTree<PODInterval<T, UserData>, PODIntervalNodeUpdater> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(PODIntervalTree);
public:
    using IntervalType = PODInterval<T, UserData>;
    class OverlapsSearchAdapter;

    // Returns all intervals in the tree which overlap the given query interval, sorted by the < operator.
    Vector<IntervalType> allOverlaps(const IntervalType& interval) const
    {
        Vector<IntervalType> result;
        OverlapsSearchAdapter adapter(result, interval);
        allOverlapsWithAdapter(adapter);
        return result;
    }

    template<typename AdapterType> void allOverlapsWithAdapter(AdapterType& adapter) const
    {
        searchForOverlapsFrom(this->root(), adapter);
    }

    std::optional<IntervalType> nextIntervalAfter(const T& point)
    {
        auto next = smallestNodeGreaterThanFrom(point, this->root());
        if (!next)
            return std::nullopt;
        return next->data();
    }

#ifndef NDEBUG

    bool checkInvariants() const
    {
        if (!Base::checkInvariants())
            return false;
        if (!this->root())
            return true;
        return checkInvariantsFromNode(this->root(), nullptr);
    }

#endif

private:
    using Base = PODRedBlackTree<PODInterval<T, UserData>, PODIntervalNodeUpdater>;
    using IntervalNode = typename Base::Node;

    // Starting from the given node, adds all overlaps with the given
    // interval to the result vector. The intervals are sorted by
    // increasing low endpoint.
    template<typename AdapterType> void searchForOverlapsFrom(IntervalNode* node, AdapterType& adapter) const
    {
        if (!node)
            return;

        // Because the intervals are sorted by left endpoint, inorder
        // traversal produces results sorted as desired.

        // See whether we need to traverse the left subtree.
        IntervalNode* left = node->left();
        if (left
            // This is phrased this way to avoid the need for operator
            // <= on type T.
            && !(left->data().maxHigh() < adapter.lowValue()))
            searchForOverlapsFrom<AdapterType>(left, adapter);

        // Check for overlap with current node.
        adapter.collectIfNeeded(node->data());

        // See whether we need to traverse the right subtree.
        // This is phrased this way to avoid the need for operator <=
        // on type T.
        if (!(adapter.highValue() < node->data().low()))
            searchForOverlapsFrom<AdapterType>(node->right(), adapter);
    }

    IntervalNode* smallestNodeGreaterThanFrom(const T& point, IntervalNode* node) const
    {
        if (!node)
            return nullptr;

        if (!(point < node->data().low()))
            return smallestNodeGreaterThanFrom(point, node->right());

        if (auto left = smallestNodeGreaterThanFrom(point, node->right()))
            return left;

        return node;
    }

#ifndef NDEBUG

    bool checkInvariantsFromNode(IntervalNode* node, T* currentMaxValue) const
    {
        // These assignments are only done in order to avoid requiring
        // a default constructor on type T.
        T leftMaxValue(node->data().maxHigh());
        T rightMaxValue(node->data().maxHigh());
        IntervalNode* left = node->left();
        IntervalNode* right = node->right();
        if (left) {
            if (!checkInvariantsFromNode(left, &leftMaxValue))
                return false;
        }
        if (right) {
            if (!checkInvariantsFromNode(right, &rightMaxValue))
                return false;
        }
        if (!left && !right) {
            // Base case.
            if (currentMaxValue)
                *currentMaxValue = node->data().high();
            return (node->data().high() == node->data().maxHigh());
        }
        T localMaxValue(node->data().maxHigh());
        if (!left || !right) {
            if (left)
                localMaxValue = leftMaxValue;
            else
                localMaxValue = rightMaxValue;
        } else
            localMaxValue = (leftMaxValue < rightMaxValue) ? rightMaxValue : leftMaxValue;
        if (localMaxValue < node->data().high())
            localMaxValue = node->data().high();
        if (!(localMaxValue == node->data().maxHigh())) {
            TextStream stream;
            stream << "localMaxValue=" << localMaxValue << "and data =" << node->data();
            LOG_ERROR("PODIntervalTree verification failed at node 0x%p: %s",
                node, stream.release().utf8().data());
            return false;
        }
        if (currentMaxValue)
            *currentMaxValue = localMaxValue;
        return true;
    }

#endif

};

#define TZONE_TEMPLATE_PARAMS template<typename T, typename UserData>
#define TZONE_TYPE PODIntervalTree<T, UserData>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

template<typename T, typename UserData> class PODIntervalTree<T, UserData>::OverlapsSearchAdapter {
public:
    using IntervalType = PODInterval<T, UserData>;

    OverlapsSearchAdapter(Vector<IntervalType>& result, const IntervalType& interval)
        : m_result(result)
        , m_interval(interval)
    {
    }

    const T& lowValue() const { return m_interval.low(); }
    const T& highValue() const { return m_interval.high(); }
    void collectIfNeeded(const IntervalType& data) const
    {
        if (data.overlaps(m_interval))
            m_result.append(data);
    }

private:
    Vector<IntervalType>& m_result;
    const IntervalType& m_interval;
};

struct PODIntervalNodeUpdater {
    template<typename Node> static bool update(Node& node)
    {
        auto* curMax = &node.data().high();
        auto* left = node.left();
        if (left) {
            if (*curMax < left->data().maxHigh())
                curMax = &left->data().maxHigh();
        }
        auto* right = node.right();
        if (right) {
            if (*curMax < right->data().maxHigh())
                curMax = &right->data().maxHigh();
        }
        // This is phrased like this to avoid needing operator!= on type T.
        if (!(*curMax == node.data().maxHigh())) {
            node.data().setMaxHigh(*curMax);
            return true;
        }
        return false;
    }
};

} // namespace WebCore
