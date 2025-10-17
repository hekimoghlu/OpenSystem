/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include "ExceptionOr.h"
#include "FetchBody.h"
#include "Supplementable.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedRawResource;
class Document;
class Navigator;
class ResourceError;

class NavigatorBeacon final : public Supplement<Navigator>, private CachedRawResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(NavigatorBeacon);
public:
    explicit NavigatorBeacon(Navigator&);
    ~NavigatorBeacon();
    static ExceptionOr<bool> sendBeacon(Navigator&, Document&, const String& url, std::optional<FetchBody::Init>&&);

    size_t inflightBeaconsCount() const { return m_inflightBeacons.size(); }

    WEBCORE_EXPORT static NavigatorBeacon* from(Navigator&);

private:
    ExceptionOr<bool> sendBeacon(Document&, const String& url, std::optional<FetchBody::Init>&&);

    static ASCIILiteral supplementName();

    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;
    void logError(const ResourceError&);

    CheckedRef<Navigator> m_navigator;
    Vector<CachedResourceHandle<CachedRawResource>> m_inflightBeacons;
};

}
