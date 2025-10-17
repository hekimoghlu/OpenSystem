/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#import "WKSnapshotConfigurationPrivate.h"

#import "WKObject.h"

@implementation WKSnapshotConfiguration {
#if PLATFORM(MAC)
    BOOL _includesSelectionHighlighting;
    BOOL _usesContentsRect;
#endif
    BOOL _usesTransparentBackground;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    self.rect = CGRectNull;
    self.afterScreenUpdates = YES;

#if PLATFORM(MAC)
    self._includesSelectionHighlighting = YES;
    self._usesContentsRect = NO;
#endif

    return self;
}

- (void)dealloc
{
    [_snapshotWidth release];

    [super dealloc];
}

- (id)copyWithZone:(NSZone *)zone
{
    WKSnapshotConfiguration *snapshotConfiguration = [(WKSnapshotConfiguration *)[[self class] allocWithZone:zone] init];

    snapshotConfiguration.rect = self.rect;
    snapshotConfiguration.snapshotWidth = self.snapshotWidth;
    snapshotConfiguration.afterScreenUpdates = self.afterScreenUpdates;

#if PLATFORM(MAC)
    snapshotConfiguration._includesSelectionHighlighting = self._includesSelectionHighlighting;
#endif
    snapshotConfiguration._usesTransparentBackground = self._usesTransparentBackground;

    return snapshotConfiguration;
}

#if PLATFORM(MAC)
- (BOOL)_includesSelectionHighlighting
{
    return _includesSelectionHighlighting;
}

- (void)_setIncludesSelectionHighlighting:(BOOL)includesSelectionHighlighting
{
    _includesSelectionHighlighting = includesSelectionHighlighting;
}

- (BOOL)_usesContentsRect
{
    return _usesContentsRect;
}

- (void)_setUsesContentsRect:(BOOL)usesContentsRect
{
    _usesContentsRect = usesContentsRect;
}

#endif // PLATFORM(MAC)

- (BOOL)_usesTransparentBackground
{
    return _usesTransparentBackground;
}

- (void)_setUsesTransparentBackground:(BOOL)usesTransparentBackground
{
    _usesTransparentBackground = usesTransparentBackground;
}

@end
