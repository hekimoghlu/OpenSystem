/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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
#import "config.h"
#import "WKContextMenuElementInfoInternal.h"

#if PLATFORM(IOS_FAMILY)

#import "_WKActivatedElementInfoInternal.h"

@implementation WKContextMenuElementInfo

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKContextMenuElementInfo.class, self))
        return;
    _elementInfo->API::ContextMenuElementInfo::~ContextMenuElementInfo();
    [super dealloc];
}

- (NSURL *)linkURL
{
    return _elementInfo->url();
}

- (API::Object&)_apiObject
{
    return *_elementInfo;
}

@end

@implementation WKContextMenuElementInfo (WKPrivate)

- (_WKActivatedElementInfo *)_activatedElementInfo
{
    return [_WKActivatedElementInfo activatedElementInfoWithInteractionInformationAtPosition:_elementInfo->interactionInformation() userInfo:_elementInfo->userInfo().get()];
}

@end

#endif

