/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#import "WKPreviewActionItemInternal.h"

#if PLATFORM(IOS_FAMILY)

@implementation WKPreviewAction
@synthesize identifier = _identifier;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
+ (instancetype)actionWithIdentifier:(NSString *)identifier title:(NSString *)title style:(UIPreviewActionStyle)style handler:(void (^)(UIPreviewAction *action, UIViewController *previewViewController))handler
{
    WKPreviewAction *action = [self actionWithTitle:title style:style handler:handler];
    action->_identifier = identifier;
    return action;
}
ALLOW_DEPRECATED_DECLARATIONS_END

- (id)copyWithZone:(NSZone *)zone
{
    WKPreviewAction *action = [super copyWithZone:zone];
    action->_identifier = self.identifier;
    return action;
}

- (void)dealloc
{
    [_identifier release];
    [super dealloc];
}

@end

#endif // PLATFORM(IOS_FAMILY)
