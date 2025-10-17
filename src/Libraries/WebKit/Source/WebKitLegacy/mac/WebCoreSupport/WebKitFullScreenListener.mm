/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#import "WebKitFullScreenListener.h"

#import <WebCore/DocumentInlines.h>
#import <WebCore/Element.h>

#if ENABLE(FULLSCREEN_API)

#import <WebCore/FullscreenManager.h>

using namespace WebCore;

@implementation WebKitFullScreenListener

- (id)initWithElement:(Element*)element
{
    if (!(self = [super init]))
        return nil;

    _element = element;
    return self;
}

- (void)webkitWillEnterFullScreen
{
    if (_element)
        _element->document().fullscreenManager().willEnterFullscreen(*_element);
}

- (void)webkitDidEnterFullScreen
{
    if (_element)
        _element->document().fullscreenManager().didEnterFullscreen();
}

- (void)webkitWillExitFullScreen
{
    if (_element)
        _element->document().fullscreenManager().willExitFullscreen();
}

- (void)webkitDidExitFullScreen
{
    if (_element)
        _element->document().fullscreenManager().didExitFullscreen();
}

@end
#endif
