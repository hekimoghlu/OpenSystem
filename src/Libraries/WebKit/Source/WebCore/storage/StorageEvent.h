/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#include "Event.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class Storage;

class StorageEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StorageEvent);
public:
    static Ref<StorageEvent> create(const AtomString& type, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea);
    static Ref<StorageEvent> createForBindings();

    struct Init : EventInit {
        String key;
        String oldValue;
        String newValue;
        String url;
        RefPtr<Storage> storageArea;
    };

    static Ref<StorageEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    virtual ~StorageEvent();

    const String& key() const { return m_key; }
    const String& oldValue() const { return m_oldValue; }
    const String& newValue() const { return m_newValue; }
    const String& url() const { return m_url; }
    Storage* storageArea() const { return m_storageArea.get(); }

    void initStorageEvent(const AtomString& type, bool canBubble, bool cancelable, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea);

    // Needed once we support init<blank>EventNS
    // void initStorageEventNS(in DOMString namespaceURI, in DOMString typeArg, in boolean canBubbleArg, in boolean cancelableArg, in DOMString keyArg, in DOMString oldValueArg, in DOMString newValueArg, in DOMString urlArg, Storage storageAreaArg);

private:
    StorageEvent();
    StorageEvent(const AtomString& type, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea);
    StorageEvent(const AtomString&, const Init&, IsTrusted);

    String m_key;
    String m_oldValue;
    String m_newValue;
    String m_url;
    RefPtr<Storage> m_storageArea;
};

} // namespace WebCore
