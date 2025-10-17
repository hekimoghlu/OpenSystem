/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
#include "InspectorLayerTreeAgent.h"

#include "InspectorDOMAgent.h"
#include "InstrumentingAgents.h"
#include "IntRect.h"
#include "PseudoElement.h"
#include "RenderChildIterator.h"
#include "RenderLayer.h"
#include "RenderLayerBacking.h"
#include "RenderLayerCompositor.h"
#include "RenderView.h"
#include <JavaScriptCore/IdentifiersFactory.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorLayerTreeAgent);

InspectorLayerTreeAgent::InspectorLayerTreeAgent(WebAgentContext& context)
    : InspectorAgentBase("LayerTree"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::LayerTreeFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::LayerTreeBackendDispatcher::create(context.backendDispatcher, this))
{
}

InspectorLayerTreeAgent::~InspectorLayerTreeAgent() = default;

void InspectorLayerTreeAgent::didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*)
{
}

void InspectorLayerTreeAgent::willDestroyFrontendAndBackend(Inspector::DisconnectReason)
{
    disable();
}

void InspectorLayerTreeAgent::reset()
{
    m_documentLayerToIdMap.clear();
    m_idToLayer.clear();
    m_pseudoElementToIdMap.clear();
    m_idToPseudoElement.clear();
    m_suppressLayerChangeEvents = false;
}

Inspector::Protocol::ErrorStringOr<void> InspectorLayerTreeAgent::enable()
{
    m_instrumentingAgents.setEnabledLayerTreeAgent(this);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorLayerTreeAgent::disable()
{
    m_instrumentingAgents.setEnabledLayerTreeAgent(nullptr);

    reset();

    return { };
}

void InspectorLayerTreeAgent::layerTreeDidChange()
{
    if (m_suppressLayerChangeEvents)
        return;

    m_suppressLayerChangeEvents = true;

    m_frontendDispatcher->layerTreeDidChange();
}

void InspectorLayerTreeAgent::renderLayerDestroyed(const RenderLayer& renderLayer)
{
    unbind(&renderLayer);
}

void InspectorLayerTreeAgent::pseudoElementDestroyed(PseudoElement& pseudoElement)
{
    unbindPseudoElement(&pseudoElement);
}

Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>>> InspectorLayerTreeAgent::layersForNode(Inspector::Protocol::DOM::NodeId nodeId)
{
    auto* node = m_instrumentingAgents.persistentDOMAgent()->nodeForId(nodeId);
    if (!node)
        return makeUnexpected("Missing node for given nodeId"_s);

    auto* renderer = node->renderer();
    if (!renderer)
        return makeUnexpected("Missing renderer of node for given nodeId"_s);

    if (!is<RenderElement>(*renderer))
        return makeUnexpected("Missing renderer of element for given nodeId"_s);

    auto layers = JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>::create();

    gatherLayersUsingRenderObjectHierarchy(downcast<RenderElement>(*renderer), layers);

    m_suppressLayerChangeEvents = false;

    return layers;
}

void InspectorLayerTreeAgent::gatherLayersUsingRenderObjectHierarchy(RenderElement& renderer, JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>& layers)
{
    if (renderer.hasLayer()) {
        gatherLayersUsingRenderLayerHierarchy(downcast<RenderLayerModelObject>(renderer).layer(), layers);
        return;
    }

    for (auto& child : childrenOfType<RenderElement>(renderer))
        gatherLayersUsingRenderObjectHierarchy(child, layers);
}

void InspectorLayerTreeAgent::gatherLayersUsingRenderLayerHierarchy(RenderLayer* renderLayer, JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>& layers)
{
    if (renderLayer->isComposited())
        layers.addItem(buildObjectForLayer(renderLayer));

    for (renderLayer = renderLayer->firstChild(); renderLayer; renderLayer = renderLayer->nextSibling())
        gatherLayersUsingRenderLayerHierarchy(renderLayer, layers);
}

Ref<Inspector::Protocol::LayerTree::Layer> InspectorLayerTreeAgent::buildObjectForLayer(RenderLayer* renderLayer)
{
    RenderObject* renderer = &renderLayer->renderer();
    RenderLayerBacking* backing = renderLayer->backing();
    Node* node = renderer->node();

    bool isReflection = renderLayer->isReflection();
    bool isGenerated = (isReflection ? renderer->parent() : renderer)->isBeforeOrAfterContent();
    bool isAnonymous = renderer->isAnonymous();

    if (renderer->isRenderView())
        node = &renderer->document();
    else if (isReflection && isGenerated)
        node = renderer->parent()->generatingElement();
    else if (isGenerated)
        node = renderer->generatingNode();
    else if (isReflection || isAnonymous)
        node = renderer->parent()->element();

    // Basic set of properties.
    auto layerObject = Inspector::Protocol::LayerTree::Layer::create()
        .setLayerId(bind(renderLayer))
        .setNodeId(idForNode(node))
        .setBounds(buildObjectForIntRect(renderer->absoluteBoundingBoxRect()))
        .setMemory(backing->backingStoreMemoryEstimate())
        .setCompositedBounds(buildObjectForIntRect(enclosingIntRect(backing->compositedBounds())))
        .setPaintCount(backing->graphicsLayer()->repaintCount())
        .release();

    if (node && node->shadowHost())
        layerObject->setIsInShadowTree(true);

    if (isReflection)
        layerObject->setIsReflection(true);

    if (isGenerated) {
        if (isReflection)
            renderer = renderer->parent();
        layerObject->setIsGeneratedContent(true);
        layerObject->setPseudoElementId(bindPseudoElement(downcast<PseudoElement>(renderer->node())));
        if (renderer->isBeforeContent())
            layerObject->setPseudoElement("before"_s);
        else if (renderer->isAfterContent())
            layerObject->setPseudoElement("after"_s);
    }

    // FIXME: RenderView is now really anonymous but don't tell about it to the frontend before making sure it can handle it.
    if (isAnonymous && !renderer->isRenderView()) {
        layerObject->setIsAnonymous(true);
        const RenderStyle& style = renderer->style();
        if (style.pseudoElementType() == PseudoId::FirstLetter)
            layerObject->setPseudoElement("first-letter"_s);
        else if (style.pseudoElementType() == PseudoId::FirstLine)
            layerObject->setPseudoElement("first-line"_s);
    }

    return layerObject;
}

Inspector::Protocol::DOM::NodeId InspectorLayerTreeAgent::idForNode(Node* node)
{
    if (!node)
        return 0;

    InspectorDOMAgent* domAgent = m_instrumentingAgents.persistentDOMAgent();
    
    auto nodeId = domAgent->boundNodeId(node);
    if (!nodeId) {
        // FIXME: <https://webkit.org/b/213499> Web Inspector: allow DOM nodes to be instrumented at any point, regardless of whether the main document has also been instrumented
        nodeId = domAgent->pushNodeToFrontend(node);
    }

    return nodeId;
}

Ref<Inspector::Protocol::LayerTree::IntRect> InspectorLayerTreeAgent::buildObjectForIntRect(const IntRect& rect)
{
    return Inspector::Protocol::LayerTree::IntRect::create()
        .setX(rect.x())
        .setY(rect.y())
        .setWidth(rect.width())
        .setHeight(rect.height())
        .release();
}

Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::LayerTree::CompositingReasons>> InspectorLayerTreeAgent::reasonsForCompositingLayer(const Inspector::Protocol::LayerTree::LayerId& layerId)
{
    const RenderLayer* renderLayer = m_idToLayer.get(layerId);

    if (!renderLayer)
        return makeUnexpected("Missing render layer for given layerId"_s);

    OptionSet<CompositingReason> reasons = renderLayer->compositor().reasonsForCompositing(*renderLayer);
    auto compositingReasons = Inspector::Protocol::LayerTree::CompositingReasons::create().release();

    if (reasons.contains(CompositingReason::Transform3D))
        compositingReasons->setTransform3D(true);

    if (reasons.contains(CompositingReason::Video))
        compositingReasons->setVideo(true);
    else if (reasons.contains(CompositingReason::Canvas))
        compositingReasons->setCanvas(true);
    else if (reasons.contains(CompositingReason::Plugin))
        compositingReasons->setPlugin(true);
    else if (reasons.contains(CompositingReason::IFrame))
        compositingReasons->setIFrame(true);
    else if (reasons.contains(CompositingReason::Model))
        compositingReasons->setModel(true);

    if (reasons.contains(CompositingReason::BackfaceVisibilityHidden))
        compositingReasons->setBackfaceVisibilityHidden(true);

    if (reasons.contains(CompositingReason::ClipsCompositingDescendants))
        compositingReasons->setClipsCompositingDescendants(true);

    if (reasons.contains(CompositingReason::Animation))
        compositingReasons->setAnimation(true);

    if (reasons.contains(CompositingReason::Filters))
        compositingReasons->setFilters(true);

    if (reasons.contains(CompositingReason::PositionFixed))
        compositingReasons->setPositionFixed(true);

    if (reasons.contains(CompositingReason::PositionSticky))
        compositingReasons->setPositionSticky(true);

    if (reasons.contains(CompositingReason::OverflowScrolling))
        compositingReasons->setOverflowScrollingTouch(true);

    // FIXME: handle OverflowScrollPositioning (webkit.org/b/195985).

    if (reasons.contains(CompositingReason::Stacking))
        compositingReasons->setStacking(true);

    if (reasons.contains(CompositingReason::Overlap))
        compositingReasons->setOverlap(true);

    if (reasons.contains(CompositingReason::NegativeZIndexChildren))
        compositingReasons->setNegativeZIndexChildren(true);

    if (reasons.contains(CompositingReason::TransformWithCompositedDescendants))
        compositingReasons->setTransformWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::OpacityWithCompositedDescendants))
        compositingReasons->setOpacityWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::MaskWithCompositedDescendants))
        compositingReasons->setMaskWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::ReflectionWithCompositedDescendants))
        compositingReasons->setReflectionWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::FilterWithCompositedDescendants))
        compositingReasons->setFilterWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::BlendingWithCompositedDescendants))
        compositingReasons->setBlendingWithCompositedDescendants(true);

    if (reasons.contains(CompositingReason::IsolatesCompositedBlendingDescendants))
        compositingReasons->setIsolatesCompositedBlendingDescendants(true);

    if (reasons.contains(CompositingReason::Perspective))
        compositingReasons->setPerspective(true);

    if (reasons.contains(CompositingReason::Preserve3D))
        compositingReasons->setPreserve3D(true);

    if (reasons.contains(CompositingReason::WillChange))
        compositingReasons->setWillChange(true);

    if (reasons.contains(CompositingReason::Root))
        compositingReasons->setRoot(true);

    if (reasons.contains(CompositingReason::BackdropRoot))
        compositingReasons->setBackdropRoot(true);

    return compositingReasons;
}

String InspectorLayerTreeAgent::bind(const RenderLayer* layer)
{
    if (!layer)
        return emptyString();
    return m_documentLayerToIdMap.ensure(layer, [this, layer] {
        auto identifier = IdentifiersFactory::createIdentifier();
        m_idToLayer.set(identifier, layer);
        return identifier;
    }).iterator->value;
}

void InspectorLayerTreeAgent::unbind(const RenderLayer* layer)
{
    auto identifier = m_documentLayerToIdMap.take(layer);
    if (identifier.isNull())
        return;
    m_idToLayer.remove(identifier);
}

String InspectorLayerTreeAgent::bindPseudoElement(PseudoElement* pseudoElement)
{
    if (!pseudoElement)
        return emptyString();
    return m_pseudoElementToIdMap.ensure(*pseudoElement, [this, pseudoElement] {
        auto identifier = IdentifiersFactory::createIdentifier();
        m_idToPseudoElement.set(identifier, pseudoElement);
        return identifier;
    }).iterator->value;
}

void InspectorLayerTreeAgent::unbindPseudoElement(PseudoElement* pseudoElement)
{
    if (!pseudoElement)
        return;
    auto identifier = m_pseudoElementToIdMap.take(*pseudoElement);
    if (identifier.isNull())
        return;
    m_idToPseudoElement.remove(identifier);
}

} // namespace WebCore
