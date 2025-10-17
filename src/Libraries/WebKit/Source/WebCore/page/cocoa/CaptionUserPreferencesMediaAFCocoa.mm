/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "CaptionUserPreferencesMediaAF.h"

#if ENABLE(VIDEO) && PLATFORM(COCOA)

#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/spi/cocoa/NSObjCRuntimeSPI.h>

@interface WebCaptionUserPreferencesMediaAFWeakObserver : NSObject {
    WeakPtr<WebCore::CaptionUserPreferencesMediaAF> m_weakPtr;
}
@property (nonatomic, readonly, direct) RefPtr<WebCore::CaptionUserPreferencesMediaAF> userPreferences;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithWeakPtr:(WeakPtr<WebCore::CaptionUserPreferencesMediaAF>&&)weakPtr NS_DESIGNATED_INITIALIZER;
@end

NS_DIRECT_MEMBERS
@implementation WebCaptionUserPreferencesMediaAFWeakObserver
- (instancetype)initWithWeakPtr:(WeakPtr<WebCore::CaptionUserPreferencesMediaAF>&&)weakPtr
{
    if ((self = [super init]))
        m_weakPtr = WTFMove(weakPtr);
    return self;
}

- (RefPtr<WebCore::CaptionUserPreferencesMediaAF>)userPreferences
{
    return m_weakPtr.get();
}
@end

namespace WebCore {

RetainPtr<WebCaptionUserPreferencesMediaAFWeakObserver> CaptionUserPreferencesMediaAF::createWeakObserver(CaptionUserPreferencesMediaAF* thisPtr)
{
    return adoptNS([[WebCaptionUserPreferencesMediaAFWeakObserver alloc] initWithWeakPtr:WeakPtr { *thisPtr }]);
}

RefPtr<CaptionUserPreferencesMediaAF> CaptionUserPreferencesMediaAF::extractCaptionUserPreferencesMediaAF(void* observer)
{
    RetainPtr strongObserver { dynamic_objc_cast<WebCaptionUserPreferencesMediaAFWeakObserver>(reinterpret_cast<id>(observer)) };
    if (!strongObserver)
        return nullptr;
    return [strongObserver userPreferences];
}

} // namespace WebCore

#endif // ENABLE(VIDEO) && PLATFORM(COCOA)
