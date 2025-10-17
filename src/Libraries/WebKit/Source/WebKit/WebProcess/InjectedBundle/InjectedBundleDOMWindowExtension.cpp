/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include "InjectedBundleDOMWindowExtension.h"

#include "InjectedBundleScriptWorld.h"
#include "WebFrame.h"
#include "WebLocalFrameLoaderClient.h"
#include <WebCore/DOMWindowExtension.h>
#include <WebCore/DOMWrapperWorld.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/LocalFrame.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

using ExtensionMap = HashMap<WeakRef<WebCore::DOMWindowExtension>, WeakRef<InjectedBundleDOMWindowExtension>>;
static ExtensionMap& allExtensions()
{
    static NeverDestroyed<ExtensionMap> map;
    return map;
}

Ref<InjectedBundleDOMWindowExtension> InjectedBundleDOMWindowExtension::create(WebFrame* frame, InjectedBundleScriptWorld* world)
{
    return adoptRef(*new InjectedBundleDOMWindowExtension(frame, world));
}

InjectedBundleDOMWindowExtension* InjectedBundleDOMWindowExtension::get(DOMWindowExtension* extension)
{
    ASSERT(allExtensions().contains(extension));
    return allExtensions().get(extension);
}

InjectedBundleDOMWindowExtension::InjectedBundleDOMWindowExtension(WebFrame* frame, InjectedBundleScriptWorld* world)
    : m_coreExtension(DOMWindowExtension::create(frame->coreLocalFrame() ? frame->coreLocalFrame()->window() : nullptr, world->coreWorld()))
{
    allExtensions().add(m_coreExtension.get(), *this);
}

InjectedBundleDOMWindowExtension::~InjectedBundleDOMWindowExtension()
{
    ASSERT(allExtensions().contains(m_coreExtension));
    allExtensions().remove(m_coreExtension);
}

RefPtr<WebFrame> InjectedBundleDOMWindowExtension::frame() const
{
    auto* frame = m_coreExtension->frame();
    if (!frame)
        return nullptr;

    return WebFrame::fromCoreFrame(*frame);
}

InjectedBundleScriptWorld* InjectedBundleDOMWindowExtension::world() const
{
    if (!m_world)
        m_world = InjectedBundleScriptWorld::getOrCreate(m_coreExtension->world());
        
    return m_world.get();
}

} // namespace WebKit
