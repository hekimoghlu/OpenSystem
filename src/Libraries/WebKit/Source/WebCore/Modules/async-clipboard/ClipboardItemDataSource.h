/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Clipboard;
class ClipboardItem;
class DeferredPromise;
class PasteboardCustomData;

class ClipboardItemDataSource {
public:
    ClipboardItemDataSource(ClipboardItem& item)
        : m_item(item)
    {
    }

    virtual ~ClipboardItemDataSource() = default;

    virtual Vector<String> types() const = 0;
    virtual void getType(const String&, Ref<DeferredPromise>&&) = 0;
    virtual void collectDataForWriting(Clipboard& destination, CompletionHandler<void(std::optional<PasteboardCustomData>)>&&) = 0;

protected:
    WeakRef<ClipboardItem> m_item;
};

} // namespace WebCore
