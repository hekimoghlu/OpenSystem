/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

#import "config.h"
#import "ContentKeyGroupFactoryAVFObjC.h"

#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)

#import "CDMKeyGroupingStrategy.h"
#import "WebAVContentKeyGroup.h"
#import <pal/graphics/cocoa/WebAVContentKeyReportGroupExtras.h>
#import <wtf/RetainPtr.h>

namespace WebCore {

RetainPtr<WebAVContentKeyGrouping> ContentKeyGroupFactoryAVFObjC::createContentKeyGroup(CDMKeyGroupingStrategy keyGroupingStrategy, AVContentKeySession *session, ContentKeyGroupDataSource& dataSource)
{
    switch (keyGroupingStrategy) {
    case CDMKeyGroupingStrategy::Platform:
#if HAVE(AVCONTENTKEYREPORTGROUP)
        return [session makeContentKeyGroup];
#else
        return nil;
#endif
    case CDMKeyGroupingStrategy::BuiltIn:
        return adoptNS([[WebAVContentKeyGroup alloc] initWithContentKeySession:session dataSource:dataSource]);
    }
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
