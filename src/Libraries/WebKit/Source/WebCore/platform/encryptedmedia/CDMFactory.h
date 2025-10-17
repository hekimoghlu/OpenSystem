/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include <memory>
#include <wtf/Forward.h>

#if !RELEASE_LOG_DISABLED
namespace WTF {
class Logger;
}
#endif

namespace WebCore {

class CDMPrivate;
class CDMPrivateClient;

class CDMFactory {
public:
    virtual ~CDMFactory() { };
    virtual std::unique_ptr<CDMPrivate> createCDM(const String&, const CDMPrivateClient&) = 0;
    virtual bool supportsKeySystem(const String&) = 0;

    WEBCORE_EXPORT static Vector<CDMFactory*>& registeredFactories();
    WEBCORE_EXPORT static void registerFactory(CDMFactory&);
    WEBCORE_EXPORT static void unregisterFactory(CDMFactory&);

    // Platform-specific function that's called when the list of
    // registered CDMFactory objects is queried for the first time.
    WEBCORE_EXPORT static void platformRegisterFactories(Vector<CDMFactory*>&);
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
