/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#include "ExceptionOr.h"
#include "LocalDOMWindowProperty.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class Node;
class Position;
class Range;
class StaticRange;
class VisibleSelection;

struct SimpleRange;

class DOMSelection : public RefCounted<DOMSelection>, public LocalDOMWindowProperty {
public:
    static Ref<DOMSelection> create(LocalDOMWindow&);

    RefPtr<Node> baseNode() const;
    RefPtr<Node> extentNode() const;
    unsigned baseOffset() const;
    unsigned extentOffset() const;
    String type() const;
    String direction() const;
    ExceptionOr<void> setBaseAndExtent(Node* baseNode, unsigned baseOffset, Node* extentNode, unsigned extentOffset);
    ExceptionOr<void> setPosition(Node*, unsigned offset);
    void modify(const String& alter, const String& direction, const String& granularity);

    // The anchor and focus are the start and end of the selection, and
    // reflect the direction in which the selection was made by the user.
    // The base and extent are different, because they don't reflect expansion.
    RefPtr<Node> anchorNode() const;
    unsigned anchorOffset() const;
    RefPtr<Node> focusNode() const;
    unsigned focusOffset() const;
    bool isCollapsed() const;
    unsigned rangeCount() const;
    ExceptionOr<void> collapse(Node*, unsigned offset);
    ExceptionOr<void> collapseToEnd();
    ExceptionOr<void> collapseToStart();
    ExceptionOr<void> extend(Node&, unsigned offset);
    ExceptionOr<Ref<Range>> getRangeAt(unsigned);
    void removeAllRanges();
    void addRange(Range&);
    ExceptionOr<void> removeRange(Range&);

    Vector<Ref<StaticRange>> getComposedRanges(FixedVector<std::reference_wrapper<ShadowRoot>>&&);

    void deleteFromDocument();
    bool containsNode(Node&, bool partlyContained) const;
    ExceptionOr<void> selectAllChildren(Node&);

    String toString() const;

    void empty();

private:
    explicit DOMSelection(LocalDOMWindow&);

    // FIXME: Change LocalDOMWindowProperty::frame to return RefPtr and then delete this.
    RefPtr<LocalFrame> frame() const;
    std::optional<SimpleRange> range() const;

    Position anchorPosition() const;
    Position focusPosition() const;
    Position basePosition() const;
    Position extentPosition() const;

    RefPtr<Node> shadowAdjustedNode(const Position&) const;
    unsigned shadowAdjustedOffset(const Position&) const;

    bool isValidForPosition(Node*) const;
};

} // namespace WebCore
