/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

#include "InspectorWebAgentBase.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IntRect;
class Node;
class PseudoElement;
class RenderElement;
class RenderLayer;
class WeakPtrImplWithEventTargetData;

class InspectorLayerTreeAgent final : public InspectorAgentBase, public Inspector::LayerTreeBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorLayerTreeAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorLayerTreeAgent);
public:
    InspectorLayerTreeAgent(WebAgentContext&);
    ~InspectorLayerTreeAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // LayerTreeBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>>> layersForNode(Inspector::Protocol::DOM::NodeId);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::LayerTree::CompositingReasons>> reasonsForCompositingLayer(const Inspector::Protocol::LayerTree::LayerId&);

    // InspectorInstrumentation
    void layerTreeDidChange();
    void renderLayerDestroyed(const RenderLayer&);
    void pseudoElementDestroyed(PseudoElement&);
    void reset();

private:
    // RenderLayer-related methods.
    String bind(const RenderLayer*);
    void unbind(const RenderLayer*);

    void gatherLayersUsingRenderObjectHierarchy(RenderElement&, JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>&);
    void gatherLayersUsingRenderLayerHierarchy(RenderLayer*, JSON::ArrayOf<Inspector::Protocol::LayerTree::Layer>&);

    Ref<Inspector::Protocol::LayerTree::Layer> buildObjectForLayer(RenderLayer*);
    Ref<Inspector::Protocol::LayerTree::IntRect> buildObjectForIntRect(const IntRect&);

    Inspector::Protocol::DOM::NodeId idForNode(Node*);

    String bindPseudoElement(PseudoElement*);
    void unbindPseudoElement(PseudoElement*);

    std::unique_ptr<Inspector::LayerTreeFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::LayerTreeBackendDispatcher> m_backendDispatcher;

    HashMap<const RenderLayer*, Inspector::Protocol::LayerTree::LayerId> m_documentLayerToIdMap;
    HashMap<Inspector::Protocol::LayerTree::LayerId, const RenderLayer*> m_idToLayer;

    WeakHashMap<PseudoElement, Inspector::Protocol::LayerTree::PseudoElementId, WeakPtrImplWithEventTargetData> m_pseudoElementToIdMap;
    HashMap<Inspector::Protocol::LayerTree::PseudoElementId, WeakPtr<PseudoElement, WeakPtrImplWithEventTargetData>> m_idToPseudoElement;

    bool m_suppressLayerChangeEvents { false };
};

} // namespace WebCore
