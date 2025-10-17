/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include <wtf/Forward.h>
#include <wtf/RefCounted.h>

namespace PAL {
class SessionID;
}

namespace WebCore {

class Page;
class SecurityOrigin;
class StorageArea;

class StorageNamespace : public RefCounted<StorageNamespace> {
public:
    virtual ~StorageNamespace() = default;
    virtual Ref<StorageArea> storageArea(const SecurityOrigin&) = 0;
    virtual const SecurityOrigin* topLevelOrigin() const = 0;

    // FIXME: This is only valid for session storage and should probably be moved to a subclass.
    virtual Ref<StorageNamespace> copy(Page& newPage) = 0;

    virtual PAL::SessionID sessionID() const = 0;
    virtual void setSessionIDForTesting(PAL::SessionID) = 0;

    virtual uint64_t storageAreaMapCountForTesting() const { return 0; }
};

} // namespace WebCore
