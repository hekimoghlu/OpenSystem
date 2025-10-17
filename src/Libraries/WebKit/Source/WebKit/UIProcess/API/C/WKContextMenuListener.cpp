/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "WKContextMenuListener.h"

#include "APIArray.h"
#include "WKAPICast.h"
#include "WebContextMenuItem.h"
#include "WebContextMenuListenerProxy.h"

using namespace WebKit;

WKTypeID WKContextMenuListenerGetTypeID()
{
#if ENABLE(CONTEXT_MENUS)
    return toAPI(WebContextMenuListenerProxy::APIType);
#else
    return toAPI(API::Object::Type::Null);
#endif
}

void WKContextMenuListenerUseContextMenuItems(WKContextMenuListenerRef listenerRef, WKArrayRef arrayRef)
{
#if ENABLE(CONTEXT_MENUS)
    RefPtr<API::Array> array = toImpl(arrayRef);
    size_t newSize = array ? array->size() : 0;
    Vector<Ref<WebContextMenuItem>> items;
    items.reserveInitialCapacity(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        WebContextMenuItem* item = array->at<WebContextMenuItem>(i);
        if (!item)
            continue;
        
        items.append(*item);
    }

    toImpl(listenerRef)->useContextMenuItems(WTFMove(items));
#else
    UNUSED_PARAM(listenerRef);
    UNUSED_PARAM(arrayRef);
#endif
}
