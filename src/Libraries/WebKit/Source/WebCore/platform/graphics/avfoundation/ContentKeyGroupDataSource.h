/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
// FIXME (116158267): This file can be removed and its implementation merged directly into
// CDMInstanceSessionFairPlayStreamingAVFObjC once we no logner need to support a configuration
// where the BuiltInCDMKeyGroupingStrategyEnabled preference is off.

#pragma once

#if HAVE(AVCONTENTKEYSESSION)

#include <wtf/Assertions.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVContentKey;

namespace WebCore {
class ContentKeyGroupDataSource;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ContentKeyGroupDataSource> : std::true_type { };
}

namespace WebCore {

class ContentKeyGroupDataSource : public CanMakeWeakPtr<ContentKeyGroupDataSource> {
public:
    virtual ~ContentKeyGroupDataSource() = default;

    virtual Vector<RetainPtr<AVContentKey>> contentKeyGroupDataSourceKeys() const = 0;
#if !RELEASE_LOG_DISABLED
    virtual uint64_t contentKeyGroupDataSourceLogIdentifier() const = 0;
    virtual const Logger& contentKeyGroupDataSourceLogger() const = 0;
    virtual WTFLogChannel& contentKeyGroupDataSourceLogChannel() const = 0;
#endif // !RELEASE_LOG_DISABLED
};

} // namespace WebCore

#endif // HAVE(AVCONTENTKEYSESSION)
