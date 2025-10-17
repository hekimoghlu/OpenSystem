/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#include "HTMLFrameOwnerElement.h"
#include "Image.h"
#include "JSValueInWrappedObject.h"
#include "RenderEmbeddedObject.h"

namespace JSC {
namespace Bindings {
class Instance;
}
}

namespace WebCore {

class PluginReplacement;
class PluginViewBase;
class RenderWidget;
class VoidCallback;

class HTMLPlugInElement : public HTMLFrameOwnerElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLPlugInElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLPlugInElement);
public:
    virtual ~HTMLPlugInElement();

    void resetInstance();

    JSC::Bindings::Instance* bindingsInstance();

    enum class PluginLoadingPolicy { DoNotLoad, Load };
    WEBCORE_EXPORT PluginViewBase* pluginWidget(PluginLoadingPolicy = PluginLoadingPolicy::Load) const;

    enum DisplayState {
        Playing,
        PreparingPluginReplacement,
        DisplayingPluginReplacement,
    };
    DisplayState displayState() const { return m_displayState; }
    void setDisplayState(DisplayState);

    bool isCapturingMouseEvents() const { return m_isCapturingMouseEvents; }
    void setIsCapturingMouseEvents(bool capturing) { m_isCapturingMouseEvents = capturing; }

#if PLATFORM(IOS_FAMILY)
    bool willRespondToMouseMoveEvents() const final { return false; }
#endif
    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    virtual bool isPlugInImageElement() const = 0;

    WEBCORE_EXPORT void pluginDestroyedWithPendingPDFTestCallback(RefPtr<VoidCallback>&&);
    WEBCORE_EXPORT RefPtr<VoidCallback> takePendingPDFTestCallback();

protected:
    HTMLPlugInElement(const QualifiedName& tagName, Document&, OptionSet<TypeFlag> = { });

    bool canContainRangeEndPoint() const override { return false; }
    void willDetachRenderers() override;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const override;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) override;

    virtual bool useFallbackContent() const { return false; }

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode& parentOfInsertedTree) override;
    void removedFromAncestor(RemovalType, ContainerNode& oldParentOfRemovedTree) override;

    void defaultEventHandler(Event&) final;

    virtual bool requestObject(const String& url, const String& mimeType, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool isReplaced(const RenderStyle&) const final;
    void didAddUserAgentShadowRoot(ShadowRoot&) final;

    // This will load the plugin if necessary.
    virtual RenderWidget* renderWidgetLoadingPlugin() const;

private:
    void swapRendererTimerFired();
    bool shouldOverridePlugin(const String& url, const String& mimeType);

    bool supportsFocus() const final;

    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isPluginElement() const final;
    bool canLoadScriptURL(const URL&) const final;

    RefPtr<JSC::Bindings::Instance> m_instance;
    Timer m_swapRendererTimer;
    RefPtr<PluginReplacement> m_pluginReplacement;
    bool m_isCapturingMouseEvents { false };
    DisplayState m_displayState { Playing };

    RefPtr<VoidCallback> m_pendingPDFTestCallback;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLPlugInElement)
    static bool isType(const WebCore::Node& node) { return node.isPluginElement(); }
SPECIALIZE_TYPE_TRAITS_END()
