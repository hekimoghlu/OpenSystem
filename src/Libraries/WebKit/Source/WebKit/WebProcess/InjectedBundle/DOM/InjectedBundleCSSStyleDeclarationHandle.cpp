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
#include "config.h"
#include "InjectedBundleCSSStyleDeclarationHandle.h"

#include <JavaScriptCore/APICast.h>
#include <WebCore/CSSStyleDeclaration.h>
#include <WebCore/JSCSSStyleDeclaration.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakRef.h>

namespace WebKit {
using namespace WebCore;

using DOMStyleDeclarationHandleCache = HashMap<SingleThreadWeakRef<CSSStyleDeclaration>, WeakRef<InjectedBundleCSSStyleDeclarationHandle>>;

static DOMStyleDeclarationHandleCache& domStyleDeclarationHandleCache()
{
    static NeverDestroyed<DOMStyleDeclarationHandleCache> cache;
    return cache;
}

RefPtr<InjectedBundleCSSStyleDeclarationHandle> InjectedBundleCSSStyleDeclarationHandle::getOrCreate(JSContextRef, JSObjectRef object)
{
    CSSStyleDeclaration* cssStyleDeclaration = JSCSSStyleDeclaration::toWrapped(toJS(object)->vm(), toJS(object));
    return getOrCreate(cssStyleDeclaration);
}

RefPtr<InjectedBundleCSSStyleDeclarationHandle> InjectedBundleCSSStyleDeclarationHandle::getOrCreate(CSSStyleDeclaration* styleDeclaration)
{
    if (!styleDeclaration)
        return nullptr;

    RefPtr<InjectedBundleCSSStyleDeclarationHandle> newHandle;
    auto result = domStyleDeclarationHandleCache().ensure(*styleDeclaration, [&] {
        newHandle = adoptRef(*new InjectedBundleCSSStyleDeclarationHandle(*styleDeclaration));
        return WeakRef { *newHandle };
    });
    return newHandle ? newHandle.releaseNonNull() : Ref { result.iterator->value.get() };
}

InjectedBundleCSSStyleDeclarationHandle::InjectedBundleCSSStyleDeclarationHandle(CSSStyleDeclaration& styleDeclaration)
    : m_styleDeclaration(styleDeclaration)
{
}

InjectedBundleCSSStyleDeclarationHandle::~InjectedBundleCSSStyleDeclarationHandle()
{
    domStyleDeclarationHandleCache().remove(m_styleDeclaration.get());
}

CSSStyleDeclaration* InjectedBundleCSSStyleDeclarationHandle::coreCSSStyleDeclaration()
{
    return m_styleDeclaration.ptr();
}

} // namespace WebKit
