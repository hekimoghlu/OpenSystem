/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#import "config.h"
#import "InspectorExtensionDelegate.h"

#if ENABLE(INSPECTOR_EXTENSIONS)

#import "APIFrameHandle.h"
#import "WebInspectorUIProxy.h"
#import "_WKFrameHandleInternal.h"
#import "_WKInspectorExtensionDelegate.h"
#import "_WKInspectorExtensionInternal.h"
#import <wtf/TZoneMallocInlines.h>
#import <wtf/UniqueRef.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorExtensionDelegate);

InspectorExtensionDelegate::InspectorExtensionDelegate(_WKInspectorExtension *inspectorExtension, id <_WKInspectorExtensionDelegate> delegate)
    : m_inspectorExtension(inspectorExtension)
    , m_delegate(delegate)
{
    m_delegateMethods.inspectorExtensionDidShowTabWithIdentifier = [delegate respondsToSelector:@selector(inspectorExtension:didShowTabWithIdentifier:withFrameHandle:)];
    m_delegateMethods.inspectorExtensionDidHideTabWithIdentifier = [delegate respondsToSelector:@selector(inspectorExtension:didHideTabWithIdentifier:)];
    m_delegateMethods.inspectorExtensionDidNavigateTabWithIdentifier = [delegate respondsToSelector:@selector(inspectorExtension:didNavigateTabWithIdentifier:newURL:)];
    m_delegateMethods.inspectorExtensionInspectedPageDidNavigate = [delegate respondsToSelector:@selector(inspectorExtension:inspectedPageDidNavigate:)];

    inspectorExtension->_extension->setClient(makeUniqueRef<InspectorExtensionClient>(*this));
}

InspectorExtensionDelegate::~InspectorExtensionDelegate() = default;

RetainPtr<id <_WKInspectorExtensionDelegate>> InspectorExtensionDelegate::delegate()
{
    return m_delegate.get();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorExtensionDelegate::InspectorExtensionClient);

InspectorExtensionDelegate::InspectorExtensionClient::InspectorExtensionClient(InspectorExtensionDelegate& delegate)
    : m_inspectorExtensionDelegate(delegate)
{
}

InspectorExtensionDelegate::InspectorExtensionClient::~InspectorExtensionClient()
{
}

void InspectorExtensionDelegate::InspectorExtensionClient::didShowExtensionTab(const Inspector::ExtensionTabID& extensionTabID, WebCore::FrameIdentifier frameID)
{
    if (!m_inspectorExtensionDelegate->m_delegateMethods.inspectorExtensionDidShowTabWithIdentifier)
        return;

    auto& delegate = m_inspectorExtensionDelegate->m_delegate;
    if (!delegate)
        return;

    [delegate inspectorExtension:m_inspectorExtensionDelegate->m_inspectorExtension.get().get() didShowTabWithIdentifier:extensionTabID withFrameHandle:wrapper(API::FrameHandle::create(frameID)).get()];
}

void InspectorExtensionDelegate::InspectorExtensionClient::didHideExtensionTab(const Inspector::ExtensionTabID& extensionTabID)
{
    if (!m_inspectorExtensionDelegate->m_delegateMethods.inspectorExtensionDidHideTabWithIdentifier)
        return;

    auto& delegate = m_inspectorExtensionDelegate->m_delegate;
    if (!delegate)
        return;

    [delegate inspectorExtension:m_inspectorExtensionDelegate->m_inspectorExtension.get().get() didHideTabWithIdentifier:extensionTabID];
}

void InspectorExtensionDelegate::InspectorExtensionClient::didNavigateExtensionTab(const Inspector::ExtensionTabID& extensionTabID, const WTF::URL& newURL)
{
    if (!m_inspectorExtensionDelegate->m_delegateMethods.inspectorExtensionDidNavigateTabWithIdentifier)
        return;

    auto& delegate = m_inspectorExtensionDelegate->m_delegate;
    if (!delegate)
        return;

    [delegate inspectorExtension:m_inspectorExtensionDelegate->m_inspectorExtension.get().get() didNavigateTabWithIdentifier:extensionTabID newURL:newURL];
}

void InspectorExtensionDelegate::InspectorExtensionClient::inspectedPageDidNavigate(const WTF::URL& newURL)
{
    if (!m_inspectorExtensionDelegate->m_delegateMethods.inspectorExtensionInspectedPageDidNavigate)
        return;

    auto& delegate = m_inspectorExtensionDelegate->m_delegate;
    if (!delegate)
        return;

    [delegate inspectorExtension:m_inspectorExtensionDelegate->m_inspectorExtension.get().get() inspectedPageDidNavigate:newURL];
}

} // namespace WebKit

#endif // ENABLE(INSPECTOR_EXTENSIONS)
