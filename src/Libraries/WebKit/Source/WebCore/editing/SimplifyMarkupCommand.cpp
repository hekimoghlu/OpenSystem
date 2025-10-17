/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#include "SimplifyMarkupCommand.h"

#include "NodeRenderStyle.h"
#include "NodeTraversal.h"
#include "RenderInline.h"
#include "RenderObject.h"
#include "RenderStyle.h"

namespace WebCore {

SimplifyMarkupCommand::SimplifyMarkupCommand(Ref<Document>&& document, Node* firstNode, Node* nodeAfterLast)
    : CompositeEditCommand(WTFMove(document))
    , m_firstNode(firstNode)
    , m_nodeAfterLast(nodeAfterLast)
{
}
    
void SimplifyMarkupCommand::doApply()
{
    RefPtr rootNode = m_firstNode->parentNode();
    Vector<Ref<Node>> nodesToRemove;
    
    protectedDocument()->updateLayoutIgnorePendingStylesheets();

    // Walk through the inserted nodes, to see if there are elements that could be removed
    // without affecting the style. The goal is to produce leaner markup even when starting
    // from a verbose fragment.
    // We look at inline elements as well as non top level divs that don't have attributes. 
    for (RefPtr node = m_firstNode.get(); node && node != m_nodeAfterLast; node = NodeTraversal::next(*node)) {
        if (node->firstChild() || (node->isTextNode() && node->nextSibling()) || !node->parentNode())
            continue;
        
        RefPtr startingNode = node->parentNode();
        auto* startingStyle = startingNode->renderStyle();
        if (!startingStyle)
            continue;
        RefPtr currentNode = startingNode;
        RefPtr<Node> topNodeWithStartingStyle;
        while (currentNode != rootNode) {
            // FIXME: The simplification algorithm should be rewritten to eliminate redundant
            // parents in cases where the children affect rendered content, as observed with
            // <span><picture></picture></span>.
            if (currentNode->hasTagName(HTMLNames::pictureTag))
                break;

            if (currentNode->parentNode() != rootNode && isRemovableBlock(currentNode.get()))
                nodesToRemove.append(*currentNode);
            
            currentNode = currentNode->parentNode();
            if (!currentNode)
                break;

            CheckedPtr renderInline = dynamicDowncast<RenderInline>(currentNode->renderer());
            if (!renderInline || renderInline->mayAffectLayout())
                continue;
            
            if (currentNode->firstChild() != currentNode->lastChild()) {
                topNodeWithStartingStyle = nullptr;
                break;
            }
            
            OptionSet<StyleDifferenceContextSensitiveProperty> contextSensitiveProperties;
            if (currentNode->renderStyle()->diff(*startingStyle, contextSensitiveProperties) == StyleDifference::Equal)
                topNodeWithStartingStyle = currentNode;
            
        }
        if (topNodeWithStartingStyle) {
            for (RefPtr node = startingNode; node && node != topNodeWithStartingStyle; node = node->parentNode())
                nodesToRemove.append(*node);
        }
    }

    // we perform all the DOM mutations at once.
    for (size_t i = 0; i < nodesToRemove.size(); ++i) {
        // FIXME: We can do better by directly moving children from nodesToRemove[i].
        int numPrunedAncestors = pruneSubsequentAncestorsToRemove(nodesToRemove, i);
        if (numPrunedAncestors < 0)
            continue;
        removeNodePreservingChildren(nodesToRemove[i], AssumeContentIsAlwaysEditable);
        i += numPrunedAncestors;
    }
}

int SimplifyMarkupCommand::pruneSubsequentAncestorsToRemove(Vector<Ref<Node>>& nodesToRemove, size_t startNodeIndex)
{
    size_t pastLastNodeToRemove = startNodeIndex + 1;
    for (; pastLastNodeToRemove < nodesToRemove.size(); ++pastLastNodeToRemove) {
        if (nodesToRemove[pastLastNodeToRemove - 1]->parentNode() != nodesToRemove[pastLastNodeToRemove].ptr())
            break;
        if (nodesToRemove[pastLastNodeToRemove]->firstChild() != nodesToRemove[pastLastNodeToRemove]->lastChild())
            break;
    }

    Ref highestAncestorToRemove = nodesToRemove[pastLastNodeToRemove - 1].get();
    RefPtr parent = highestAncestorToRemove->parentNode();
    if (!parent) // Parent has already been removed.
        return -1;
    
    if (pastLastNodeToRemove == startNodeIndex + 1)
        return 0;

    removeNode(nodesToRemove[startNodeIndex], AssumeContentIsAlwaysEditable);
    insertNodeBefore(nodesToRemove[startNodeIndex].copyRef(), highestAncestorToRemove, AssumeContentIsAlwaysEditable);
    removeNode(highestAncestorToRemove, AssumeContentIsAlwaysEditable);

    return pastLastNodeToRemove - startNodeIndex - 1;
}

} // namespace WebCore
