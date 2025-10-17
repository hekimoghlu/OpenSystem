/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#import "WebOpenPanelResultListener.h"

#import <WebCore/FileChooser.h>
#import <wtf/RefPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/Icon.h>
#endif

using namespace WebCore;

@implementation WebOpenPanelResultListener

- (id)initWithChooser:(FileChooser&)chooser
{
    self = [super init];
    if (!self)
        return nil;
    _chooser = &chooser;
    return self;
}

- (void)cancel
{
    _chooser = nullptr;
}

- (void)chooseFilename:(NSString *)filename
{
    ASSERT(_chooser);
    if (!_chooser)
        return;
    _chooser->chooseFile(filename);
    _chooser = nullptr;
}

- (void)chooseFilenames:(NSArray *)filenames
{
    ASSERT(_chooser);
    if (!_chooser)
        return;
    _chooser->chooseFiles(makeVector<String>(filenames));
    _chooser = nullptr;
}

#if PLATFORM(IOS_FAMILY)

- (void)chooseFilename:(NSString *)filename displayString:(NSString *)displayString iconImage:(CGImageRef)imageRef
{
    [self chooseFilenames:@[filename] displayString:displayString iconImage:imageRef];
}

- (void)chooseFilenames:(NSArray *)filenames displayString:(NSString *)displayString iconImage:(CGImageRef)imageRef
{
    ASSERT(_chooser);
    if (!_chooser)
        return;

    _chooser->chooseMediaFiles(makeVector<String>(filenames), displayString, Icon::create(imageRef).get());
    _chooser = nullptr;
}

#endif

@end
