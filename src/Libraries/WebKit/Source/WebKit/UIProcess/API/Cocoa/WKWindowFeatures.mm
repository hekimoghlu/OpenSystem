/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#import "WKWindowFeaturesInternal.h"

#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WindowFeatures.h>
#import <wtf/RetainPtr.h>

@implementation WKWindowFeatures

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWindowFeatures.class, self))
        return;

    _windowFeatures->API::WindowFeatures::~WindowFeatures();

    [super dealloc];
}

- (NSNumber *)menuBarVisibility
{
    if (auto menuBarVisible = _windowFeatures->windowFeatures().menuBarVisible)
        return @(*menuBarVisible);

    return nil;
}

- (NSNumber *)statusBarVisibility
{
    if (auto statusBarVisible = _windowFeatures->windowFeatures().statusBarVisible)
        return @(*statusBarVisible);

    return nil;
}

- (NSNumber *)toolbarsVisibility
{
    if (auto toolBarVisible = _windowFeatures->windowFeatures().toolBarVisible)
        return @(*toolBarVisible);

    return nil;
}

- (NSNumber *)allowsResizing
{
    if (auto resizable = _windowFeatures->windowFeatures().resizable)
        return @(*resizable);

    return nil;
}

- (NSNumber *)x
{
    if (auto x = _windowFeatures->windowFeatures().x)
        return @(*x);

    return nil;
}

- (NSNumber *)y
{
    if (auto y = _windowFeatures->windowFeatures().y)
        return @(*y);

    return nil;
}

- (NSNumber *)width
{
    if (auto width = _windowFeatures->windowFeatures().width)
        return @(*width);

    return nil;
}

- (NSNumber *)height
{
    if (auto height = _windowFeatures->windowFeatures().height)
        return @(*height);

    return nil;
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_windowFeatures;
}

@end

@implementation WKWindowFeatures (WKPrivate)

- (BOOL)_wantsPopup
{
    return _windowFeatures->windowFeatures().wantsPopup();
}

- (BOOL)_hasAdditionalFeatures
{
    return _windowFeatures->windowFeatures().hasAdditionalFeatures;
}

- (NSNumber *)_popup
{
    if (auto popup = _windowFeatures->windowFeatures().popup)
        return @(*popup);

    return nil;
}

- (NSNumber *)_locationBarVisibility
{
    if (auto locationBarVisible = _windowFeatures->windowFeatures().locationBarVisible)
        return @(*locationBarVisible);

    return nil;
}

- (NSNumber *)_scrollbarsVisibility
{
    if (auto scrollbarsVisible = _windowFeatures->windowFeatures().scrollbarsVisible)
        return @(*scrollbarsVisible);

    return nil;
}

- (NSNumber *)_fullscreenDisplay
{
    if (auto fullscreen = _windowFeatures->windowFeatures().fullscreen)
        return @(*fullscreen);

    return nil;
}

- (NSNumber *)_dialogDisplay
{
    if (auto dialog = _windowFeatures->windowFeatures().dialog)
        return @(*dialog);

    return nil;
}

@end
