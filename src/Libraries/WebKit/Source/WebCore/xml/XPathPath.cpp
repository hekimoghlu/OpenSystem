/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "XPathPath.h"

#include "Document.h"
#include "XPathPredicate.h"
#include "XPathStep.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace XPath {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Filter);
WTF_MAKE_TZONE_ALLOCATED_IMPL(LocationPath);
WTF_MAKE_TZONE_ALLOCATED_IMPL(Path);

        
Filter::Filter(std::unique_ptr<Expression> expression, Vector<std::unique_ptr<Expression>> predicates)
    : m_expression(WTFMove(expression)), m_predicates(WTFMove(predicates))
{
    setIsContextNodeSensitive(m_expression->isContextNodeSensitive());
    setIsContextPositionSensitive(m_expression->isContextPositionSensitive());
    setIsContextSizeSensitive(m_expression->isContextSizeSensitive());
}

Value Filter::evaluate() const
{
    Value result = m_expression->evaluate();
    
    NodeSet& nodes = result.modifiableNodeSet();
    nodes.sort();

    EvaluationContext& evaluationContext = Expression::evaluationContext();
    for (auto& predicate : m_predicates) {
        NodeSet newNodes;
        evaluationContext.size = nodes.size();
        evaluationContext.position = 0;
        
        for (auto& node : nodes) {
            evaluationContext.node = node;
            ++evaluationContext.position;
            
            if (evaluatePredicate(*predicate))
                newNodes.append(node.copyRef());
        }
        nodes = WTFMove(newNodes);
    }

    return result;
}

LocationPath::LocationPath()
    : m_isAbsolute(false)
{
    setIsContextNodeSensitive(true);
}

Value LocationPath::evaluate() const
{
    EvaluationContext& evaluationContext = Expression::evaluationContext();
    EvaluationContext backupContext = evaluationContext;
    // http://www.w3.org/TR/xpath/
    // Section 2, Location Paths:
    // "/ selects the document root (which is always the parent of the document element)"
    // "A / by itself selects the root node of the document containing the context node."
    // In the case of a tree that is detached from the document, we violate
    // the spec and treat / as the root node of the detached tree.
    // This is for compatibility with Firefox, and also seems like a more
    // logical treatment of where you would expect the "root" to be.
    Node* context = evaluationContext.node.get();
    if (m_isAbsolute && !context->isDocumentNode())
        context = &context->rootNode();

    NodeSet nodes;
    nodes.append(context);
    evaluate(nodes);
    
    evaluationContext = backupContext;
    return Value(WTFMove(nodes));
}

void LocationPath::evaluate(NodeSet& nodes) const
{
    bool resultIsSorted = nodes.isSorted();

    for (auto& step : m_steps) {
        NodeSet newNodes;
        HashSet<RefPtr<Node>> newNodesSet;

        bool needToCheckForDuplicateNodes = !nodes.subtreesAreDisjoint() || (step->axis() != Step::ChildAxis && step->axis() != Step::SelfAxis
            && step->axis() != Step::DescendantAxis && step->axis() != Step::DescendantOrSelfAxis && step->axis() != Step::AttributeAxis);

        if (needToCheckForDuplicateNodes)
            resultIsSorted = false;

        // This is a simplified check that can be improved to handle more cases.
        if (nodes.subtreesAreDisjoint() && (step->axis() == Step::ChildAxis || step->axis() == Step::SelfAxis))
            newNodes.markSubtreesDisjoint(true);

        for (auto& node : nodes) {
            NodeSet matches;
            step->evaluate(*node, matches);

            if (!matches.isSorted())
                resultIsSorted = false;

            for (auto& match : matches) {
                if (!needToCheckForDuplicateNodes || newNodesSet.add(match.get()).isNewEntry)
                    newNodes.append(match.copyRef());
            }
        }
        
        nodes = WTFMove(newNodes);
    }

    nodes.markSorted(resultIsSorted);
}

void LocationPath::appendStep(std::unique_ptr<Step> step)
{
    unsigned stepCount = m_steps.size();
    if (stepCount) {
        bool dropSecondStep;
        optimizeStepPair(*m_steps[stepCount - 1], *step, dropSecondStep);
        if (dropSecondStep)
            return;
    }
    step->optimize();
    m_steps.append(WTFMove(step));
}

void LocationPath::prependStep(std::unique_ptr<Step> step)
{
    if (m_steps.size()) {
        bool dropSecondStep;
        optimizeStepPair(*step, *m_steps[0], dropSecondStep);
        if (dropSecondStep) {
            m_steps[0] = WTFMove(step);
            return;
        }
    }
    step->optimize();
    m_steps.insert(0, WTFMove(step));
}

Path::Path(std::unique_ptr<Expression> filter, std::unique_ptr<LocationPath> path)
    : m_filter(WTFMove(filter))
    , m_path(WTFMove(path))
{
    setIsContextNodeSensitive(m_filter->isContextNodeSensitive());
    setIsContextPositionSensitive(m_filter->isContextPositionSensitive());
    setIsContextSizeSensitive(m_filter->isContextSizeSensitive());
}

Value Path::evaluate() const
{
    Value result = m_filter->evaluate();

    NodeSet& nodes = result.modifiableNodeSet();
    m_path->evaluate(nodes);
    
    return result;
}

}
}
