/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

#include "StorageSessionProvider.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/SchedulePair.h>
#endif

#if PLATFORM(COCOA)
OBJC_CLASS NSOperationQueue;
#endif

#if USE(SOUP)
typedef struct _SoupSession SoupSession;
#endif

namespace WebCore {

class NetworkStorageSession;
class ResourceError;
class ResourceRequest;

class NetworkingContext : public StorageSessionProvider {
public:
    virtual ~NetworkingContext() = default;

    virtual bool isValid() const { return true; }

    virtual bool shouldClearReferrerOnHTTPSToHTTPRedirect() const = 0;

#if PLATFORM(COCOA)
    virtual bool localFileContentSniffingEnabled() const = 0; // FIXME: Reconcile with ResourceHandle::forceContentSniffing().
    virtual SchedulePairHashSet* scheduledRunLoopPairs() const { return 0; }
    virtual RetainPtr<CFDataRef> sourceApplicationAuditData() const = 0;
    virtual ResourceError blockedError(const ResourceRequest&) const = 0;
#endif

    virtual String sourceApplicationIdentifier() const { return emptyString(); }

#if PLATFORM(WIN)
    virtual ResourceError blockedError(const ResourceRequest&) const = 0;
#endif

protected:
    NetworkingContext() = default;
};

}
